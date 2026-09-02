"""Pre-extract frozen SPMM text-encoder embeddings into a .npz cache.

One file per (checkpoint, target, split); the training scripts then only load
arrays. Cached alongside the embedding are the 9 animal PK features and the
regression target, both of which depend on the split, so they cannot be shared.
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import SMILESDataset_single_PBPK, cv_data_transform
from xbert import BertConfig, BertForMaskedLM

DEFAULT_CACHE_DIR = './feat_cache'
CACHE_FORMAT = 1                      # bump to invalidate every cache on disk
SPLITS = ('train', 'valid', 'all')

ENCODER_KIND = 'text'
ENCODER_SPEC = 'text[l0-5] CLS'       # pins the forward path, not just its name

# the columns SMILESDataset_single_PBPK turns into `animal`, in its order
ANIMAL_COLS = ["monkey_VDss", "monkey_CL", "monkey_fup",
               "dog_VDss", "dog_CL", "dog_fup",
               "rat_VDss", "rat_CL", "rat_fup"]

DEFAULT_BERT_CONFIG_TEXT = './src/config/config_bert.json'


class TextEncoder(nn.Module):
    """SMILES branch of SPMM: mode='text' runs layers 0-5 only, since layers
    6-11 are cross-attention and need a second modality."""
    kind = ENCODER_KIND
    spec = ENCODER_SPEC

    def __init__(self, config=None):
        super().__init__()
        bert_config = BertConfig.from_json_file(config['bert_config_text'])

        self.text_encoder = BertForMaskedLM(config=bert_config)
        self.text_encoder.cls = nn.Identity()
        self.text_width = self.text_encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, text_input_ids, text_attention_mask, prop=None):
        text_embed = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text').last_hidden_state[:, 0, :]
        return text_embed             # (B, text_width)


def build_encoder(checkpoint, config, device):
    """Frozen TextEncoder with `checkpoint` loaded (strict=False)."""
    encoder = TextEncoder(config=config)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
        for key in list(state_dict.keys()):
            if '_unk' in key:
                state_dict[key.replace('_unk', '_mask')] = state_dict.pop(key)
        encoder.load_state_dict(state_dict, strict=False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder.to(device)


def build_tokenizer(vocab_filename):
    tokenizer = BertTokenizer(vocab_file=vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token,
                                                       max_input_chars_per_word=250)
    return tokenizer


@torch.no_grad()
def extract_features(encoder, dataset, tokenizer, device, batch_size, max_len, desc=''):
    """One frozen pass over `dataset` -> (feat, animal, value) float32 arrays in dataframe order."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                        pin_memory=True, shuffle=False, drop_last=False)
    feats, animals, values = [], [], []
    for text, prop, animal, value in tqdm(loader, desc=f'extract {desc}'):
        text_input = tokenizer(text, padding='longest', truncation=True,
                               max_length=max_len, return_tensors="pt").to(device)
        f = encoder(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:], prop.to(device))
        feats.append(f.float().cpu())
        animals.append(animal)
        values.append(value)
    return (torch.cat(feats).numpy().astype(np.float32),
            torch.cat(animals).numpy().astype(np.float32),
            torch.cat(values).numpy().astype(np.float32))


def load_splits(input_name, target_name, imputer=False):
    """The exact train/valid dataframes the training scripts build."""
    df = pd.read_csv(f'{input_name}')
    df = df.dropna(subset=target_name).reset_index(drop=True)

    train_df = df[df[f'{target_name[6:]}_set'] == 'Train']
    valid_df = df[df[f'{target_name[6:]}_set'] == 'Valid']

    train_df, valid_df = cv_data_transform(train_df, valid_df, target_name, imputer, seed=None)
    return {'train': train_df, 'valid': valid_df}


def pool_transform(df, target_name):
    """cv_data_transform minus the RobustScaler: log10 on CL/VDss, fup linear.

    The scaler is what makes the ordinary cache split-dependent, so the pooled
    cache must hold unscaled animal values for k-fold to refit per fold.
    """
    df = df.copy()
    log_cols = [c for c in df.columns if ('_CL' in c or '_VDss' in c)]
    df[log_cols] = np.log10(df[log_cols].clip(lower=1e-12))
    return df


def load_pool(input_name, target_name):
    """{'all': df} -- every row with an observed target, no Train/Valid filtering."""
    df = pd.read_csv(f'{input_name}')
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    return {'all': pool_transform(df, target_name)}


def split_fingerprint(df, target_name):
    """sha1 over exactly the columns the cached arrays are derived from."""
    cols = ['mol'] + ANIMAL_COLS + [target_name]
    payload = df[cols].to_csv(index=False, float_format='%.10g')
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _slug(s):
    return re.sub(r'[^0-9A-Za-z._+-]', '-', str(s))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def cache_filename(encoder, checkpoint, target_name, split):
    return f'{encoder}_{_slug(_stem(checkpoint))}_{_slug(target_name)}_{split}.npz'


def cache_path(cache_dir, encoder, checkpoint, target_name, split):
    return os.path.join(cache_dir, cache_filename(encoder, checkpoint, target_name, split))


def cache_meta(checkpoint, input_name, target_name, split, max_len, vocab_filename, fingerprint):
    """Everything a loader must agree on before it trusts the arrays.

    `input`, `max_len` and `vocab` are no longer in the filename, so they are
    checked here instead: a cache written under different settings fails to
    validate rather than being silently reused.
    """
    return {
        'format': CACHE_FORMAT,
        'encoder': ENCODER_KIND,
        'encoder_spec': ENCODER_SPEC,
        'checkpoint': os.path.basename(checkpoint),
        'input': input_name,
        'target_name': target_name,
        'split': split,
        'max_len': int(max_len),
        'vocab': os.path.basename(vocab_filename),
        'fingerprint': fingerprint,
    }


def save_cache(path, feat, animal, value, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp.npz'                       # atomic: never leave a half-written cache
    np.savez(tmp, feat=feat, animal=animal, value=value, meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def read_cache(path, expected_meta):
    """(feat, animal, value) if `path` matches `expected_meta`; raises ValueError if not."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta'].item()))
        bad = [k for k, v in expected_meta.items() if meta.get(k) != v]
        if bad:
            detail = ', '.join(f'{k}: cached={meta.get(k)!r} expected={expected_meta[k]!r}' for k in bad)
            raise ValueError(f'stale cache {path} ({detail})')
        feat, animal, value = z['feat'], z['animal'], z['value']
    if not (len(feat) == len(animal) == len(value)):
        raise ValueError(f'corrupt cache {path}: row counts differ '
                         f'({len(feat)}/{len(animal)}/{len(value)})')
    return feat, animal, value


def cache_is_current(path, expected_meta):
    if not os.path.exists(path):
        return False
    try:
        read_cache(path, expected_meta)
        return True
    except (ValueError, OSError, KeyError):
        return False


def extract_command(checkpoint, input_name, target_name, max_len, vocab_filename, cache_dir,
                    pool=False):
    """The command line that would produce the missing cache file."""
    cmd = (f'python new_extract_features.py --checkpoints {checkpoint} '
           f'--input {input_name} --target_name {target_name} --max_len {max_len}')
    if pool:
        cmd += ' --pool'
    if os.path.basename(vocab_filename) != 'new_vocab_spe_496.txt':
        cmd += f' --vocab_filename {vocab_filename}'
    if os.path.abspath(cache_dir) != os.path.abspath(DEFAULT_CACHE_DIR):
        cmd += f' --cache_dir {cache_dir}'
    return cmd


def load_cached_features(encoder, checkpoint, args, splits, target_name):
    """{split: (feat, animal, value)} tensors for one checkpoint, straight off disk.

    Every file is verified against the dataframe it is paired with, so a stale
    cache fails loudly instead of silently training on the wrong rows.
    """
    out = {}
    for split, df in splits.items():
        path = cache_path(args.cache_dir, encoder, checkpoint, target_name, split)
        expected = cache_meta(checkpoint, args.input, target_name, split,
                              args.max_len, args.vocab_filename, split_fingerprint(df, target_name))
        how = extract_command(checkpoint, args.input, target_name,
                              args.max_len, args.vocab_filename, args.cache_dir,
                              pool=(split == 'all'))
        if not os.path.exists(path):
            raise SystemExit(f'[cache] missing: {path}\n'
                             f'[cache] extract it first:\n    {how}')
        try:
            feat, animal, value = read_cache(path, expected)
        except ValueError as e:
            raise SystemExit(f'[cache] {e}\n'
                             f'[cache] re-extract it:\n    {how} --overwrite')
        if len(feat) != len(df):
            raise SystemExit(f'[cache] {path} has {len(feat)} rows but the {split} split has {len(df)}\n'
                             f'[cache] re-extract it:\n    {how} --overwrite')
        out[split] = (torch.from_numpy(feat), torch.from_numpy(animal), torch.from_numpy(value))
    return out


def main(args):
    device = torch.device(args.device)
    config = {'bert_config_text': args.bert_config_text}
    tokenizer = build_tokenizer(args.vocab_filename)

    splits_per_target = {}
    for target_name in args.target_name:
        try:
            splits = (load_pool(args.input, target_name) if args.pool
                      else load_splits(args.input, target_name, args.imputer))
        except KeyError:
            # e.g. src-iwata-pk.csv has CL_set/VDss_set but no fup_set, so
            # human_fup cannot be split at all. Skip it rather than kill the run.
            continue
        splits_per_target[target_name] = {s: df for s, df in splits.items() if s in args.splits}

    for checkpoint in args.checkpoints:
        todo = []
        for target_name, splits in splits_per_target.items():
            for split, df in splits.items():
                path = cache_path(args.cache_dir, ENCODER_KIND, checkpoint, target_name, split)
                meta = cache_meta(checkpoint, args.input, target_name, split,
                                  args.max_len, args.vocab_filename,
                                  split_fingerprint(df, target_name))
                if args.overwrite or not cache_is_current(path, meta):
                    todo.append((target_name, split, df, path, meta))
        if not todo:
            continue

        encoder = build_encoder(checkpoint, config, device)
        for target_name, split, df, path, meta in todo:
            dataset = SMILESDataset_single_PBPK(df, target_name)
            feat, animal, value = extract_features(
                encoder, dataset, tokenizer, device, args.extract_batch_size,
                args.max_len, desc=f'{target_name}/{split}')
            assert len(feat) == len(df), f'{len(feat)} extracted vs {len(df)} rows'
            save_cache(path, feat, animal, value, meta)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def list_cache(cache_dir):
    if not os.path.isdir(cache_dir):
        print(f'no cache directory at {cache_dir}')
        return
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith('.npz'))
    if not files:
        print(f'{cache_dir} is empty')
        return
    total = 0
    for f in files:
        path = os.path.join(cache_dir, f)
        size = os.path.getsize(path)
        total += size
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['meta'].item()))
                shape = z['feat'].shape
            print(f'{f}\n    {shape[0]} x {shape[1]}  {size/1e6:.1f} MB  '
                  f'fingerprint={meta["fingerprint"][:12]}')
        except Exception as e:                       # listing must not die on one bad file
            print(f'{f}\n    [unreadable] {e}')
    print(f'\n{len(files)} file(s), {total/1e6:.1f} MB in {cache_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+',
                        default=['./Pretrain/trimodal_pubchem100m_96_step=173008.ckpt',
                                 './Pretrain/trimodal_pubchem100m_96_step=216260.ckpt'],
                        help='One cache file per checkpoint -- ensemble members never share.')
    parser.add_argument('--input', default='src-iwata-pk', type=str)
    parser.add_argument('--target_name', nargs='+', default=['human_CL', 'human_VDss', 'human_fup'])
    parser.add_argument('--splits', nargs='+', default=['train', 'valid'], choices=list(SPLITS))
    parser.add_argument('--pool', action='store_true',
                        help="ignore the Train/Valid columns and cache one split-independent "
                             "'all' file per (checkpoint, target). Animal values are stored "
                             'UNSCALED so k-fold can refit the scaler per fold.')
    parser.add_argument('--imputer', action='store_true',
                        help='passed through to cv_data_transform (which currently ignores it), '
                             'so it is NOT part of the cache key')
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    parser.add_argument('--max_len', default=100, type=int, help='SMILES token truncation length')
    parser.add_argument('--extract_batch_size', default=64, type=int)
    parser.add_argument('--bert_config_text', default=DEFAULT_BERT_CONFIG_TEXT)
    parser.add_argument('--cache_dir', default=DEFAULT_CACHE_DIR, type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-extract even when an up-to-date cache exists')
    parser.add_argument('--list', action='store_true', help='list the cache and exit')

    args = parser.parse_args()
    if args.pool:
        args.splits = ['all']
    if args.list:
        list_cache(args.cache_dir)
    else:
        main(args)
