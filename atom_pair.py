import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
import pickle
from calc_property import calculate_property
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

ALLOWED_ATOMS = {"C", "N", "O", "F", "S", "Cl", "Br", "H", "P", "Si", "I", "B", "Se"}
def check_atom(smiles):
    mol =Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            print(f'NOT ALLOWED ATOMS: {atom.GetSymbol()} in {smiles}')
    return smiles

def get_vocab( smi ):
    m = get_geom_rdkit( smi , mode ='precise')
    if m is None:
        print(f'!!!!!!!!!!! Failed case for get_vocab {smi} {m}')
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    vocab_list = []
    for bond in m.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom, j_atom = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
        vocab_key = f'{i_atom}-{j_atom}'
        vocab_list.append(vocab_key)
    return vocab_list

def get_geom_rdkit( smi , max_try=5, mode='normal'):
    mol = Chem.AddHs( Chem.MolFromSmiles(smi) )
    if mol is None:
        return None
    num_try= 0
    while num_try < max_try:
        try:
            if mode == 'precise':
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.useSmallRingTorsions = True
                params.useExpTorsionAnglePrefs = True
                params.useBasicKnowledge = True
                params.pruneRmsThresh = 0.5
                params.maxAttempts = 1000
                params.randomSeed = 42
            
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=25, params=params)
                if not conf_ids:
                    print('failed embedding')
                    raise RuntimeError(f"Embedding failed for {smi}")
                if not AllChem.MMFFHasAllMoleculeParams(mol):
                    print('failed mmff94 params')
                    raise RuntimeError(f"MMFF94s parameters unavailable for {smi}")
            
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                best_conf_id = None
                min_energy = float('inf')
                for cid in conf_ids:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                    ff.Minimize(maxIts=2000)
                    energy = ff.CalcEnergy()
                    if energy < min_energy:
                        min_energy = energy
                        best_conf_id = cid
            
                if best_conf_id is None:
                    print('failed search best conf id')
                    raise RuntimeError(f"All conformers failed for {smi}")
            
                best_conf = mol.GetConformer(best_conf_id)
                best_conf_cp = Chem.Conformer(best_conf)
                mol.RemoveAllConformers()
                mol.AddConformer(best_conf_cp, assignId=True)
                return mol
            else:
                AllChem.EmbedMolecule( mol ) 
                AllChem.MMFFOptimizeMolecule( mol )
                return mol

        except Exception as e:
            num_try+=1
            continue
    print(f'Failed to get geo max_try {max_try}: {smi}')

def get_dist( smi ):
    m = get_geom_rdkit( smi )
    if m is None:
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    vocab_list = []
    dist_list  = []
    for bond in m.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom, j_atom = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
        dist = dist_mat[i,j]
        vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}', atom_pair_index['[UNK]']) #assign UNK when the atom-pair excluded in vocab
        vocab_list.append(vocab_idx)
        dist_list.append(dist)
    return torch.tensor(vocab_list).long(), torch.tensor(dist_list).float()

def safe_get_dist(args):
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    idx, smi = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False, canonical=True)
        prop = ( calculate_property(smi) - property_mean) / property_std
        dist = get_dist(smi)
        return smiles, prop, dist
    except Exception as e:
        print(f'Failed {idx}: {e}')
        return None

if __name__ == '__main__':
    import time
    import glob
    file_names = glob.glob('pbpk_db/src-seal*')
    all_smiles = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        try:
            list_smiles = df['smiles']
        except:
            list_smiles = df['smiles_r']
        all_smiles.extend(list_smiles)
    tmp = [ check_atom(smiles) for smiles in all_smiles ]    
    print(f'{len(all_smiles)} -> {len(tmp)}')
    exit(-1)
    vocab = pickle.load(open('bk_atom_pair_vocab.pkl', 'rb'))
    new_vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
    idx = 0
    for item in list(vocab.keys())[4:]:
        if any(atom not in ALLOWED_ATOMS for atom in item.split('-')):
            print('failed pass: ', item)
        else:
            new_vocab[item]=idx+4
            idx+=1
    pickle.dump(new_vocab, open('atom_pair_vocab.pkl', 'wb'))
    exit(-1)

    list_smi = [line.strip() for line in open('./data/1_Pretrain/pretrain_50m.txt')][1000000:1001000]
    st = time.time()
    with Pool(24) as p:
        results = list(tqdm(p.map(get_vocab, enumerate(list_smi) ), total=len(list_smi)))
    print('end of get_vocab')
    et = time.time()
    results = [item for item in results if item is not None]
    print(f'Time for get geo:  {et-st:.2f}, {len(results)}')
    exit(-1)
    print(vocab_dict)

    df = pd.DataFrame(output, columns=['atom_pair', 'dist'])
    print(df)
    print(atom_pair_dict)
    vocab_idx = [ output[idx][0] for idx in range(len(output))]
    print(vocab_idx)
    vocab_idx = [ atom_pair_dict.get(output[idx][0]) for idx in range(len(output))]
    print(vocab_idx)
    df['idx'] = vocab_idx
    print(df)
    exit(-1)
