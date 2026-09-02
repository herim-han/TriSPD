from torch.utils.data import Dataset
import random
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import rdkit
from rdkit.Chem import AllChem, Descriptors
import torch
import pandas as pd
import pickle
from collections import Counter, OrderedDict
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import signal
class TimeoutError(Exception):
    pass

RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')
multiprocessing.set_start_method("spawn", force=True)

with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

atom_pair_index = pickle.load(open('atom_pair_vocab.pkl', 'rb') )
property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
property_mean = property_mean.detach().cpu().numpy().astype(np.float32)
property_std  = property_std.detach().cpu().numpy().astype(np.float32)

def data_preprocess(args):
    idx, smiles = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)
        prop = calculate_property(smiles)
        properties = (prop - property_mean) / property_std #numpy, numpy, numpy
        atom_list, dist_mat = get_dist(smiles)
        if dist_mat is None:
            print(f'Failed get 3DDistanceMat {idx}')
            return None
        return properties, smiles, atom_list, dist_mat

    except Exception as e:
        print(f"Failed processing {idx}: {e}")
        return None
        
def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        output.append(descriptor_dict[descriptor](mol))
    return output

def get_dist( smi ):
    m = get_geom_rdkit( smi ) #type(output) = mol
    if m is None:
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    atom_list = np.array([ atom.GetSymbol() for atom in m.GetAtoms() ])
    return atom_list, dist_mat.astype(np.float32)

def get_geom_rdkit( smi , max_try=5, mode='normal'):
    mol = Chem.AddHs( Chem.MolFromSmiles(smi) )
    params = AllChem.ETKDGv3()
    if mol is None:
        return None

    num_try= 0
    while num_try < max_try:
        try:
            AllChem.EmbedMolecule( mol, params )
            AllChem.MMFFOptimizeMolecule( mol )
            return mol

        except Exception as e:
            num_try+=1
            continue
    print(f'Failed to get geo max_try: {smi}')

if __name__=='__main__':
    import os
    import time
    import lmdb
    import glob
    list_smiles = [line.strip() for line in open(f'./test_smiles').readlines()]
    print(f'Start Sampled SMILES : {len(list_smiles)}')
    st = time.time()
    try:
        with Pool(24) as p:
            results = list(tqdm(p.imap_unordered(data_preprocess, enumerate(list_smiles), chunksize=1), total=len(list_smiles)))
        filtered_results = [item for item in results
                           if item is not None and all(x is not None for x in item) ]
        with open(f'./test_dataset.pkl', 'wb') as f:
            pickle.dump(filtered_results, f)
    except Exception as e:
        print(f'Error occur : {name}, {e}')
    et = time.time()
    print(f'Time : {et-st:.3f} / Success case: {len(filtered_results)} ')
