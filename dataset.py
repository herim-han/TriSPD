from torch.utils.data import Dataset, IterableDataset
import torch
import random
import pandas as pd
from rdkit import Chem
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
from atom_pair import get_dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import os
import lmdb, pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
import io
RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')
property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
property_mean = property_mean.detach().cpu().numpy().astype(np.float32)
property_std  = property_std.detach().cpu().numpy().astype(np.float32)
atom_vocab = pickle.load(open('atom_vocab.pkl', 'rb'))
atom_pair_vocab = pickle.load(open('atom_pair_vocab.pkl', 'rb') )

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)

def pickle_loads_compat(data):
    return NumpyCompatUnpickler(io.BytesIO(data)).load()

class ShuffledMultiLMDBDataset(Dataset):
    def __init__(self, file_names, idx_file, strip_salt_frags=True):
        self.file_names = file_names
        self.indices = np.load(idx_file) 
        self._envs = None 
        self.strip_salt_frags = strip_salt_frags

    def _init_envs(self):
        self._envs = [
            lmdb.open(f, subdir=False, readonly=True,
                      lock=False, readahead=False, meminit=False)
            for f in self.file_names
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._envs is None:
            self._init_envs()
        chunk_id, local_idx = self.indices[idx]
        key = f'{int(local_idx):012d}'.encode('ascii')
        with self._envs[int(chunk_id)].begin(write=False) as txn:
            value = txn.get(key)
            properties, smiles, atom_pair, dist = pickle.loads(value)
        if self.strip_salt_frags:
            properties, smiles, atom_pair, dist = strip_salt(properties, smiles, atom_pair, dist)
        return properties, smiles, atom_pair, dist

class LMDBDataset(Dataset):
    def __init__(self, val_lmdb_path):
        self.path = val_lmdb_path
        env = lmdb.open(val_lmdb_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            self._length = txn.stat()['entries']
        env.close()
        self._env = None

    def _init_env(self):
        self._env = lmdb.open(
            self.path, subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._env is None:
            self._init_env()
        key = f'{idx:012d}'.encode('ascii')
        with self._env.begin(write=False) as txn:
            value = txn.get(key)
            properties, smiles, atom_pair, dist = pickle_loads_compat(value)
        return properties, smiles, atom_pair, dist

def cv_data_transform(train_df, valid_df, target_name, imputer=False, seed=None):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    feat_cols = [  
            "monkey_VDss","monkey_CL","monkey_fup",
            "dog_VDss", "dog_CL","dog_fup",
            "rat_VDss", "rat_CL","rat_fup"]
    pk_params = ["_CL", "_VDss"]
    scaler = RobustScaler()
    log_scaled_col = [name for name in train_df.columns if any(pk_param in name for pk_param in pk_params)]
    train_df[log_scaled_col] = np.log10(train_df[log_scaled_col].clip(lower=1e-12))
    valid_df[log_scaled_col] = np.log10(valid_df[log_scaled_col].clip(lower=1e-12))
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    valid_df[feat_cols] = scaler.transform(valid_df[feat_cols])
    return train_df, valid_df

class SMILESDataset_single_PBPK(Dataset):
    def __init__(self, data, target_name, data_length=None, shuffle=False):
        self.data = [data.iloc[i] for i in range(len(data))]
        self.target_name = target_name
        self.feat_cols = [  
                "monkey_VDss","monkey_CL","monkey_fup",
                "dog_VDss", "dog_CL","dog_fup",
                "rat_VDss", "rat_CL","rat_fup"]
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['mol'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        prop = calculate_property(smiles)
        prop = (prop.detach().cpu().numpy().astype(np.float32) - property_mean) / property_std
        prop = prop.astype(np.float32)
        animal = np.array([float(self.data[index][c]) for c in self.feat_cols], dtype=np.float32)
        target = np.array(self.data[index][self.target_name], dtype=np.float32)
        return '[CLS]' + smiles, prop, animal, target

def to_numpy(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).tolist()
    return x

def data_preprocess(args):
    idx, smiles = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
        prop = calculate_property(smiles)
        properties = (prop.detach().cpu().numpy().astype(np.float32) - property_mean) / property_std
        properties = properties.astype(np.float32).tolist()
        atom_set, dist = get_dist(smiles) or (None, None)
        atom_set = to_numpy(atom_set)
        dist = to_numpy(dist)
        return properties, '[CLS]'+smiles, atom_set, dist

    except Exception as e:
        print(f"Failed processing {idx}: {e}")
        return None

def collate_fn(batch):
    properties, smiles, atom_lists, dist_mats = zip(*batch)
    properties = torch.tensor(np.array(properties), dtype=torch.float32)
    return properties, smiles

def tri_collate_fn(batch):
    properties, smiles, atom_lists, dist_mats = zip(*batch)                   
    properties = torch.tensor(np.array(properties), dtype=torch.float32)
    smiles = ['[CLS]'+smi for smi in smiles]
    b_atom_pairs = []
    b_dist_mat   = []
    for atom_list, dist in zip(atom_lists, dist_mats):
        dist = torch.tensor(dist, dtype=torch.float32)
        N= dist.shape[0]

        i, j = torch.tril_indices(N, N, offset=-1)
        dist_ij = dist[i, j]
        filter_i = i[dist_ij<=2.5]
        filter_j = j[dist_ij<=2.5]
        filter_dist = dist_ij[dist_ij<=2.5]

        sample_vocab_idx = []
        for i, j in zip(filter_i, filter_j):
            a1 = atom_list[int(i)]
            a2 = atom_list[int(j)]
            key = "-".join(sorted([a1, a2])) 
            idx = atom_pair_vocab.get(key, atom_pair_vocab['[UNK]'])
            sample_vocab_idx.append(idx)
        b_atom_pairs.append(sample_vocab_idx)
        b_dist_mat.append(filter_dist)
    atom_list= pad_sequence([torch.as_tensor(a, dtype=torch.long) for a in b_atom_pairs], batch_first=True, padding_value=0)
    dist_mat = pad_sequence([torch.as_tensor(d, dtype=torch.float32) for d in b_dist_mat], batch_first=True, padding_value=0)
    return properties, smiles, atom_list, dist_mat

def pad_dist_mats(dist_mats, pad_value=0.0, dtype=torch.float32):
    B = len(dist_mats)
    N_max = max(d.shape[0] for d in dist_mats) 

    padded = torch.full((B, N_max, N_max), pad_value, dtype=dtype)

    for i, d in enumerate(dist_mats):
        d = torch.as_tensor(d, dtype=dtype)
        N = d.shape[0]
        padded[i, :N, :N] = d 
    return padded
