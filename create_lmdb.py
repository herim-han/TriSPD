import lmdb, pickle, time, gc
from multiprocessing import Pool
from tqdm import tqdm
import os
from rdkit import Chem
import numpy as np
import pandas as pd
from gen_3d import data_preprocess
import lmdb
import pickle
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

def write_lmdb_from_smiles(smiles_file, lmdb_path, n_proc=24, commit_interval=50000, map_size=int(5e11)):
    list_smiles = [line.strip() for line in open(smiles_file)]
    print(f"Loaded {len(list_smiles)} SMILES")

    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False)
    txn = env.begin(write=True)
    success_count = 0
    with Pool(n_proc) as p:
        results = p.imap_unordered(data_preprocess, enumerate(list_smiles), chunksize=10)

        for idx, item in tqdm(enumerate(results), total=len(list_smiles)):
            if item is None or any(x is None for x in item):
                continue

            key = f"{success_count:012d}".encode("ascii")
            value = pickle.dumps(item, protocol=4)
            txn.put(key, value)
            
            success_count += 1
            if idx % commit_interval == 0 and idx > 0:
                txn.commit()
                txn = env.begin(write=True)
                print(f"Committed {idx} samples")
                gc.collect()

    txn.commit()
    env.close()
    print(f'Success case: {success_count}')


if __name__=="__main__":
    import os
    import subprocess
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    import glob
    import time

    write_lmdb_from_smiles(
        smiles_file=args.input_file,
        lmdb_path=args.output_file,
        n_proc=24,
        commit_interval=50000)
