"""Pre-extract frozen SPMM embeddings into a .npz cache, once per ensemble member.

d_regression_frozen_v{1,2,3}.py re-run the frozen encoder on every launch. This
script does that pass once and writes the result to disk; the new_* training
scripts then only load it:

    encoder='fusion'  <- new_d_regression_frozen_v1.py   (SMILES + property, layers 0-11)
    encoder='text'    <- new_d_regression_frozen_v2.py   (SMILES only,       layers 0-5 )
                      <- new_d_regression_frozen_v3.py   (same cache as v2; drops `animal`)

`fusion_dist` / `fusion_dist_raw` are the same two paths with the property branch
replaced by the 3D-geometry branch, so `--encoder fusion_dist` feeds
new_d_regression_frozen_v1.py unchanged. They need `pbpk_db/{input}-dist.pkl`
(build it with new_dist_source.py) and a TRIMODAL checkpoint -- the bimodal ones
have no dist weights at all.

One .npz per (encoder, checkpoint, input csv, target, split), because all three
cached arrays depend on that whole tuple:

    feat    : the frozen embedding      -> encoder + checkpoint + SMILES + max_len + vocab
    animal  : the 9 animal PK features  -> cv_data_transform's RobustScaler is fit on the
                                           TRAIN rows of THIS target, so it is split- and
                                           target-dependent, not just molecule-dependent
    value   : the regression target     -> target

Ensemble members must therefore never share a file: the two checkpoints differ only
in their `step=` suffix, which is why the checkpoint basename (not `tri`/`bi`) is in
the filename.

`--imputer` is deliberately NOT part of the key: cv_data_transform() accepts the
flag but never reads it (dataset.py), so keying on it would only duplicate caches.

Row order is the dataframe order -- extraction uses shuffle=False, drop_last=False --
and `meta.fingerprint` (sha1 over mol + animal columns + target column) is checked at
load time, so a changed csv can never be silently paired with a stale cache.

Examples
--------
    # everything run_freeze.sh needs
    python new_extract_features.py \
        --encoder fusion text \
        --checkpoints Pretrain/trimodal_pubchem100m_96_step=173008.ckpt \
                      Pretrain/trimodal_pubchem100m_96_step=216260.ckpt \
                      Pretrain/bimodal_pubchem100m_96_step=173008.ckpt \
                      Pretrain/bimodal_pubchem100m_96_step=216260.ckpt \
        --target_name human_CL human_VDss --input src-iwata-pk

    python new_extract_features.py --list          # what is cached already
"""
import argparse
import hashlib
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import SMILESDataset_single_PBPK, cv_data_transform
from trimodal_bert_models_v3 import Embed3DStruct
from xbert import BertConfig, BertForMaskedLM

DEFAULT_CACHE_DIR = './feat_cache'
CACHE_FORMAT = 1                      # bump to invalidate every cache on disk
SPLITS = ('train', 'valid', 'all')

# the columns SMILESDataset_single_PBPK turns into `animal`, in its order
ANIMAL_COLS = ["monkey_VDss", "monkey_CL", "monkey_fup",
               "dog_VDss", "dog_CL", "dog_fup",
               "rat_VDss", "rat_CL", "rat_fup"]

DEFAULT_BERT_CONFIG_TEXT = './config_bert.json'
DEFAULT_BERT_CONFIG_PROPERTY = './config_bert_property.json'
DEFAULT_BERT_CONFIG_DIST = './config_bert_dist.json'

DIST_CUTOFF = 2.5                     # A -- the pair selection pretraining used
# config_bert_dist.json has max_position_embeddings=512 and the sequence is
# dist_cls + pairs, so 511 pairs is a hard architectural ceiling, not a choice.
# 9 of the 770 iwata molecules (1.2%) exceed it; the rest are untouched.
DIST_MAX_PAIR = 511
# 'head'    keep the first 511 pairs in tril (atom-index) order -- the same
#           first-N truncation --max_len applies to SMILES, and it preserves the
#           pair ordering the position embeddings were trained on.
# 'nearest' keep the 511 shortest contacts, then restore tril order -- covers the
#           whole molecule instead of amputating its high-index half.
# Both live in the encoder `spec`, so flipping this constant invalidates every
# cached dist file instead of silently mixing two conventions.
DIST_TRUNCATE = 'head'


# ============================== frozen encoders ============================== #
class FusionEncoder(nn.Module):
    """SMILES encoder + property fusion, following the PRETRAINING path.

    d_regression_frozen_v1 (and the SPMM_regressor it came from) build
    property_encoder but never call it: they feed the raw property_embed output
    straight into the text encoder's fusion layers as K/V. Pretraining does not
    do that -- `results['prop']['embeds']` is the property_encoder OUTPUT, and
    that is what every cross-attention path consumes (ITM at
    trimodal_bert_models_v3.py:487, MLM at :550, the dist analogue at :326).
    So the fusion layers were trained on a contextualised 6-layer sequence, not
    on 53 independently projected scalars.

    This encoder restores the pretraining path: property_encoder IS run.
    The `bk_fusion__*.npz` caches hold the previous (encoder-skipping) features.

    Attribute names match SPMM_regressor so the pretrained checkpoint's
    state_dict keys still line up.
    """
    kind = 'fusion'
    # part of the cache key: changing the forward path must invalidate old files
    spec = 'text[l0-5] x property_encoder(prop_cls+property_embed) -> fusion[l6-11] CLS'

    def __init__(self, config=None):
        super().__init__()
        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        bert_config2 = BertConfig.from_json_file(config['bert_config_property'])

        self.text_encoder = BertForMaskedLM(config=bert_config)
        self.text_encoder.cls = nn.Identity()
        self.text_width = self.text_encoder.config.hidden_size

        self.property_embed = nn.Linear(1, self.text_width)
        self.property_cls = nn.Parameter(torch.randn(1, 1, self.text_width))
        self.property_encoder = BertForMaskedLM(config=bert_config2).bert

    @torch.no_grad()
    def forward(self, text_input_ids, text_attention_mask, prop):
        prop = prop.to(text_input_ids.device, non_blocking=True)
        prop_embed = self.property_embed(prop.unsqueeze(2))
        properties = torch.cat([self.property_cls.expand(prop_embed.size(0), 1, -1), prop_embed], dim=1)
        # pretraining calls the property encoder with inputs_embeds only: no
        # attention_mask (every property token is real) and the default
        # mode='multi_modal', i.e. all 6 layers. Kept identical here.
        prop_embeds = self.property_encoder(inputs_embeds=properties,
                                            return_dict=True).last_hidden_state
        text_embed = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text').last_hidden_state
        prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long).to(prop_embeds.device)
        fusion_output = self.text_encoder.bert(encoder_embeds=text_embed,
                                               attention_mask=text_attention_mask,
                                               encoder_hidden_states=prop_embeds,
                                               encoder_attention_mask=prop_atts,
                                               return_dict=True,
                                               mode='fusion',
                                               ).last_hidden_state[:, 0, :]
        return fusion_output          # (B, text_width) -- the extracted embedding

    def missing_keys_of_interest(self, missing):
        return list(missing)


class TextEncoder(nn.Module):
    """Frozen text-only encoder used by d_regression_frozen_v2/v3: the SMILES
    branch of SPMM -- no property embedding, no fusion pass. NOTE mode='text'
    runs only layers 0..fusion_layer-1 (6 of 12); layers 6-11 are the property
    cross-attention layers and cannot run without a property input."""
    kind = 'text'
    spec = 'text[l0-5] CLS'

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
        return text_embed             # (B, text_width) -- the extracted embedding

    def missing_keys_of_interest(self, missing):
        return list(missing)


class FusionRawEncoder(FusionEncoder):
    """`fusion` with property_encoder SKIPPED -- the path d_regression_frozen_v1
    and SPMM_regressor have always used: raw property_embed output straight into
    the text encoder's fusion layers as K/V.

    Kept as a first-class encoder so the "does running property_encoder matter?"
    ablation is reproducible from code on any dataset, instead of surviving only
    as bk_fusion__*.npz backups of the pre-change `fusion`.
    """
    kind = 'fusion_raw'
    spec = 'text[l0-5] x raw property_embed (no property_encoder) -> fusion[l6-11] CLS'

    @torch.no_grad()
    def forward(self, text_input_ids, text_attention_mask, prop):
        prop = prop.to(text_input_ids.device, non_blocking=True)
        prop_embed = self.property_embed(prop.unsqueeze(2))
        prop_embed = torch.cat([self.property_cls.expand(prop_embed.size(0), 1, -1), prop_embed], dim=1)
        text_embed = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text').last_hidden_state
        prop_atts = torch.ones(prop_embed.size()[:-1], dtype=torch.long).to(prop_embed.device)
        fusion_output = self.text_encoder.bert(encoder_embeds=text_embed,
                                               attention_mask=text_attention_mask,
                                               encoder_hidden_states=prop_embed,
                                               encoder_attention_mask=prop_atts,
                                               return_dict=True,
                                               mode='fusion',
                                               ).last_hidden_state[:, 0, :]
        return fusion_output

    def missing_keys_of_interest(self, missing):
        # property_encoder genuinely is unused here, so a miss there is harmless
        return [k for k in missing if not k.startswith('property_encoder.')]


class FusionDistEncoder(nn.Module):
    """SMILES encoder + 3D-geometry fusion: the `fusion` path with the property
    branch swapped for the distance branch.

    Both branches feed the SAME cross-attention stack -- text_encoder layers
    6-11, mode='fusion' -- during pretraining, so the fusion weights have seen
    dist K/V as well as prop K/V (MLM at trimodal_bert_models_v3.py:319-321,
    `enriched_text` at :326). This encoder reproduces the dist half of that:

        dist_cls + dist_embed_layer(atom_pair, distance)  ->  dist_encoder (4L)
        text[l0-5]  x  that  ->  fusion[l6-11] CLS

    Pretraining passes NO attention mask on the dist side -- `atts` is
    torch.ones over the padded length (:200) -- so a padded batch lets every
    molecule attend to its neighbours' padding and the extracted feature would
    depend on batch composition. Extraction therefore runs unpadded, one
    molecule per forward; see extract_dist_features.

    Only the trimodal checkpoints carry these weights; build_encoder refuses a
    bimodal one rather than emitting random-init features.
    """
    kind = 'fusion_dist'
    needs_dist = True
    spec = ('text[l0-5] x dist_encoder(dist_cls+dist_embed_layer) -> fusion[l6-11] CLS '
            f'| pairs<={DIST_CUTOFF}A, max{DIST_MAX_PAIR}/{DIST_TRUNCATE}, unpadded')

    def __init__(self, config=None):
        super().__init__()
        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        bert_config3 = BertConfig.from_json_file(config['bert_config_dist'])

        self.text_encoder = BertForMaskedLM(config=bert_config)
        self.text_encoder.cls = nn.Identity()
        self.text_width = self.text_encoder.config.hidden_size

        # widths exactly as SPMM.__init__ builds them (trimodal_bert_models_v3.py:63-77)
        dist_width = bert_config3.hidden_size
        self.dist_embed_layer = Embed3DStruct(dist_width // 2, dist_width // 2)
        self.dist_cls = nn.Parameter(torch.randn(1, 1, dist_width))
        self.dist_encoder = BertForMaskedLM(config=bert_config3).bert

    def dist_kv(self, atom_pair, dist):
        """(B, P+1, D) key/value sequence for the fusion layers."""
        dist_feature = self.dist_embed_layer(atom_pair, dist)
        distances = torch.cat([self.dist_cls.expand(dist_feature.size(0), 1, -1), dist_feature], dim=1)
        # pretraining calls dist_encoder with inputs_embeds only: no mask, all 4 layers
        return self.dist_encoder(inputs_embeds=distances, return_dict=True).last_hidden_state

    @torch.no_grad()
    def forward(self, text_input_ids, text_attention_mask, atom_pair, dist):
        dist_embeds = self.dist_kv(atom_pair.to(text_input_ids.device),
                                   dist.to(text_input_ids.device))
        text_embed = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask,
                                            return_dict=True, mode='text').last_hidden_state
        dist_atts = torch.ones(dist_embeds.size()[:-1], dtype=torch.long).to(dist_embeds.device)
        fusion_output = self.text_encoder.bert(encoder_embeds=text_embed,
                                               attention_mask=text_attention_mask,
                                               encoder_hidden_states=dist_embeds,
                                               encoder_attention_mask=dist_atts,
                                               return_dict=True,
                                               mode='fusion',
                                               ).last_hidden_state[:, 0, :]
        return fusion_output

    def missing_keys_of_interest(self, missing):
        return list(missing)


class FusionDistRawEncoder(FusionDistEncoder):
    """`fusion_dist` with dist_encoder SKIPPED -- the dist analogue of
    `fusion_raw`, so the "does running the modality encoder matter?" ablation is
    the same question on both branches.

    K/V is then dist_cls + the raw Embed3DStruct output, i.e. 384 pair-symbol
    embedding dims concatenated with 384 Gaussian-RBF distance dims, with no
    contextualisation across pairs.
    """
    kind = 'fusion_dist_raw'
    spec = ('text[l0-5] x raw dist_embed_layer (no dist_encoder) -> fusion[l6-11] CLS '
            f'| pairs<={DIST_CUTOFF}A, max{DIST_MAX_PAIR}/{DIST_TRUNCATE}, unpadded')

    def dist_kv(self, atom_pair, dist):
        dist_feature = self.dist_embed_layer(atom_pair, dist)
        return torch.cat([self.dist_cls.expand(dist_feature.size(0), 1, -1), dist_feature], dim=1)

    def missing_keys_of_interest(self, missing):
        # dist_encoder genuinely is unused here, so a miss there is harmless
        return [k for k in missing if not k.startswith('dist_encoder.')]


ENCODERS = {'fusion': FusionEncoder, 'fusion_raw': FusionRawEncoder, 'text': TextEncoder,
            'fusion_dist': FusionDistEncoder, 'fusion_dist_raw': FusionDistRawEncoder}


def needs_dist(kind):
    return getattr(ENCODERS[kind], 'needs_dist', False)


def build_encoder(kind, checkpoint, config, device):
    encoder = ENCODERS[kind](config=config)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']
        if needs_dist(kind) and not any(k.startswith('dist_embed_layer.') for k in state_dict):
            raise SystemExit(
                f'[encoder] {kind} needs the 3D branch, but {os.path.basename(checkpoint)} has no '
                f'dist_* weights -- it is a bimodal checkpoint.\n'
                f'[encoder] use a trimodal_*.ckpt, or extract `fusion`/`fusion_raw` instead.')
        for key in list(state_dict.keys()):
            if '_unk' in key:
                state_dict[key.replace('_unk', '_mask')] = state_dict.pop(key)
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        n_all = len(encoder.state_dict())
        print(f'load checkpoint from {checkpoint} ({n_all - len(missing)}/{n_all} encoder tensors restored)')
        used_missing = encoder.missing_keys_of_interest(missing)
        if used_missing:
            print(f'  [warn] {len(used_missing)} tensors used by forward() are NOT in the checkpoint '
                  f'(e.g. {used_missing[:3]}) -- they stay randomly initialised')
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
    """Run the frozen encoder once over `dataset` -> (feats, animals, values) as
    float32 numpy arrays, in dataset (= dataframe) order."""
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


# ============================ 3D geometry -> pairs =========================== #
def pairs_within_cutoff(atoms, dist_matrix, atom_pair_vocab, cutoff=DIST_CUTOFF,
                        max_pair=DIST_MAX_PAIR, truncate=DIST_TRUNCATE):
    """(atom_pair_idx, distances) for every atom pair closer than `cutoff`.

    Byte-for-byte the selection tri_collate_fn (dataset.py:298-320) applies to
    the pretraining LMDBs: lower triangle of the full 3D distance matrix, kept
    where d <= cutoff, each pair keyed by its sorted symbol string -- then
    truncated to `max_pair`, which pretraining never needed because its
    molecules were smaller than the dist encoder's 512 position embeddings.

    Note the vocabulary was built from BONDED pairs only (atom_pair.get_vocab
    iterates GetBonds()), so through-space pairs that are never bonded -- H-H
    above all -- fall to [UNK]. That is ~17% of the iwata pairs, and it is what
    the checkpoint was pretrained with, so it is reproduced, not repaired.
    """
    d = torch.as_tensor(dist_matrix, dtype=torch.float32)
    n = d.shape[0]
    i, j = torch.tril_indices(n, n, offset=-1)
    dij = d[i, j]
    keep = dij <= cutoff
    fi, fj, fd = i[keep], j[keep], dij[keep]

    if fd.numel() > max_pair:
        if truncate == 'nearest':
            pick = torch.argsort(fd)[:max_pair].sort().values   # nearest, back in tril order
        else:
            pick = torch.arange(max_pair)
        fi, fj, fd = fi[pick], fj[pick], fd[pick]

    unk = atom_pair_vocab['[UNK]']
    idx = [atom_pair_vocab.get('-'.join(sorted([str(atoms[x]), str(atoms[y])])), unk)
           for x, y in zip(fi.tolist(), fj.tolist())]
    return torch.tensor(idx, dtype=torch.long), fd


class PBPKDistDataset(torch.utils.data.Dataset):
    """SMILESDataset_single_PBPK plus the molecule's <=2.5 A pair sequence.

    The 3D store is keyed by canonical isomeric SMILES, which is exactly what
    SMILESDataset_single_PBPK produces (dataset.py:218), so the two never drift.
    """

    def __init__(self, df, target_name, dist_store):
        self.base = SMILESDataset_single_PBPK(df, target_name)
        self.dist_store = dist_store
        self.atom_pair_vocab = pickle.load(open('atom_pair_vocab.pkl', 'rb'))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        text, prop, animal, value = self.base[i]
        smiles = text[len('[CLS]'):]
        if smiles not in self.dist_store:
            raise KeyError(f'no 3D geometry for {smiles}')
        atoms, dist_matrix = self.dist_store[smiles]
        atom_pair, dist = pairs_within_cutoff(atoms, dist_matrix, self.atom_pair_vocab)
        return text, prop, animal, value, atom_pair, dist


@torch.no_grad()
def extract_dist_features(encoder, dataset, tokenizer, device, max_len, desc=''):
    """Same contract as extract_features, but one molecule per forward.

    Batching would require padding the pair sequence, and the dist side carries
    no attention mask in pretraining (trimodal_bert_models_v3.py:200), so padded
    positions are genuinely attended to: a molecule's embedding would depend on
    which molecules shared its batch. Unpadded keeps the cache reproducible.
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=8,
                        pin_memory=True, shuffle=False, drop_last=False)
    feats, animals, values = [], [], []
    for text, prop, animal, value, atom_pair, dist in tqdm(loader, desc=f'extract {desc}'):
        text_input = tokenizer(text, padding='longest', truncation=True,
                               max_length=max_len, return_tensors="pt").to(device)
        f = encoder(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:],
                    atom_pair, dist)
        feats.append(f.float().cpu())
        animals.append(animal)
        values.append(value)
    return (torch.cat(feats).numpy().astype(np.float32),
            torch.cat(animals).numpy().astype(np.float32),
            torch.cat(values).numpy().astype(np.float32))


# ================================ data splits ================================ #
def load_splits(input_name, target_name, imputer=False):
    """The exact train/valid dataframes d_regression_frozen_v* build, so the
    extractor and the training scripts can never drift apart."""
    df = pd.read_csv(f'pbpk_db/{input_name}.csv')
    df = df.dropna(subset=target_name).reset_index(drop=True)

    train_df = df[df[f'{target_name[6:]}_set'] == 'Train']
    valid_df = df[df[f'{target_name[6:]}_set'] == 'Valid']

    train_df, valid_df = cv_data_transform(train_df, valid_df, target_name, imputer, seed=None)
    return {'train': train_df, 'valid': valid_df}


def pool_transform(df, target_name):
    """`cv_data_transform` minus the RobustScaler: log10 on every CL/VDss column
    (targets included, fup left linear), animal columns left in their own units.

    The scaler is what makes the ordinary cache split-dependent -- it is fit on
    the train rows, so a different partition produces different animal numbers
    for the same molecule. k-fold refits it per fold instead, which means the
    cached array must be the *unscaled* one.
    """
    df = df.copy()
    log_cols = [c for c in df.columns if ('_CL' in c or '_VDss' in c)]
    df[log_cols] = np.log10(df[log_cols].clip(lower=1e-12))
    return df


def load_pool(input_name, target_name):
    """{'all': df} -- every row with an observed target, no Train/Valid filtering.

    The partition is decided at training time by k-fold, so the extractor must
    not bake one in. Row order is the csv order after dropna, and the k-fold
    script indexes the cached arrays with exactly that order.
    """
    df = pd.read_csv(f'pbpk_db/{input_name}.csv')
    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    return {'all': pool_transform(df, target_name)}


def split_fingerprint(df, target_name):
    """sha1 over exactly the columns the cached arrays are derived from."""
    cols = ['mol'] + ANIMAL_COLS + [target_name]
    payload = df[cols].to_csv(index=False, float_format='%.10g')
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


# ================================ cache layout =============================== #
def _slug(s):
    return re.sub(r'[^0-9A-Za-z._+-]', '-', str(s))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def cache_filename(encoder, checkpoint, input_name, target_name, split, max_len, vocab_filename):
    return (f'{encoder}__{_slug(_stem(checkpoint))}__{_slug(input_name)}__{_slug(target_name)}'
            f'__{split}__len{max_len}__{_slug(_stem(vocab_filename))}.npz')


def cache_path(cache_dir, encoder, checkpoint, input_name, target_name, split, max_len, vocab_filename):
    return os.path.join(cache_dir, cache_filename(encoder, checkpoint, input_name, target_name,
                                                  split, max_len, vocab_filename))

def cache_meta(encoder, checkpoint, input_name, target_name, split, max_len, vocab_filename, fingerprint):
    """Everything a loader must agree on before it trusts the arrays.

    `encoder_spec` pins the forward path, not just its name: when an encoder's
    computation changes (as `fusion` did when property_encoder was put back in),
    caches written by the old path stop validating instead of being silently
    reused under the same filename.
    """
    return {
        'format': CACHE_FORMAT,
        'encoder': encoder,
        'encoder_spec': ENCODERS[encoder].spec,
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


def extract_command(encoder, checkpoint, input_name, target_name, max_len, vocab_filename, cache_dir,
                    pool=False):
    cmd = (f'python new_extract_features.py --encoder {encoder} '
           f'--checkpoints {checkpoint} --input {input_name} --target_name {target_name} '
           f'--max_len {max_len}')
    if pool:
        cmd += ' --pool'
    if os.path.basename(vocab_filename) != 'new_vocab_spe_496.txt':
        cmd += f' --vocab_filename {vocab_filename}'
    if os.path.abspath(cache_dir) != os.path.abspath(DEFAULT_CACHE_DIR):
        cmd += f' --cache_dir {cache_dir}'
    return cmd


# ========================== loader for the new_* scripts ===================== #
def load_cached_features(encoder, checkpoint, args, splits, target_name):
    """{split: (feat, animal, value)} torch tensors for one checkpoint.

    Read straight off disk -- no encoder, no tokenizer, no GPU. Every file is
    verified against the dataframe it is being paired with, so a stale cache
    fails loudly instead of silently training on the wrong rows.
    """
    out = {}
    for split, df in splits.items():
        path = cache_path(args.cache_dir, encoder, checkpoint, args.input, target_name,
                          split, args.max_len, args.vocab_filename)
        expected = cache_meta(encoder, checkpoint, args.input, target_name, split,
                              args.max_len, args.vocab_filename, split_fingerprint(df, target_name))
        how = extract_command(encoder, checkpoint, args.input, target_name,
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
        print(f'[cache] {split}: {tuple(feat.shape)} <- {path}')
        out[split] = (torch.from_numpy(feat), torch.from_numpy(animal), torch.from_numpy(value))
    return out


# =================================== main ==================================== #
def main(args):
    device = torch.device(args.device)
    config = {'bert_config_text': args.bert_config_text,
              'bert_config_property': args.bert_config_property,
              'bert_config_dist': args.bert_config_dist}
    tokenizer = build_tokenizer(args.vocab_filename)

    # the 3D store is shared by both dist encoders; load it once, and only if asked
    dist_store = None
    if any(needs_dist(k) for k in args.encoder):
        from new_dist_source import load_store
        dist_store = load_store(args.input)
        print(f'[dist] {len(dist_store)} molecule(s) with 3D geometry '
              f'<- pbpk_db/{args.input}-dist.pkl')

    # dataframes first: shared across every encoder/checkpoint, and cheap
    splits_per_target = {}
    for target_name in args.target_name:
        try:
            splits = (load_pool(args.input, target_name) if args.pool
                      else load_splits(args.input, target_name, args.imputer))
        except KeyError as e:
            # e.g. src-iwata-pk.csv has CL_set/VDss_set but no fup_set, so
            # human_fup cannot be split at all (the training scripts die the
            # same way). Skip it instead of killing the whole extraction run.
            print(f'[skip] {target_name}: {args.input}.csv has no {e} column')
            continue
        splits = {s: df for s, df in splits.items() if s in args.splits}
        splits_per_target[target_name] = splits
        sizes = ', '.join(f'{s} n={len(df)}' for s, df in splits.items())
        print(f'[data] {args.input}.csv / {target_name}: {sizes}')

    n_written = n_skipped = 0
    for encoder_kind in args.encoder:
        for checkpoint in args.checkpoints:
            # what is still missing for this (encoder, checkpoint)?
            todo = []
            for target_name, splits in splits_per_target.items():
                for split, df in splits.items():
                    path = cache_path(args.cache_dir, encoder_kind, checkpoint, args.input,
                                      target_name, split, args.max_len, args.vocab_filename)
                    meta = cache_meta(encoder_kind, checkpoint, args.input, target_name, split,
                                      args.max_len, args.vocab_filename,
                                      split_fingerprint(df, target_name))
                    if not args.overwrite and cache_is_current(path, meta):
                        print(f'[skip] up to date: {path}')
                        n_skipped += 1
                        continue
                    todo.append((target_name, split, df, path, meta))
            if not todo:
                continue

            print(f'=== extracting {encoder_kind} embeddings: {checkpoint} ===')
            encoder = build_encoder(encoder_kind, checkpoint, config, device)
            for target_name, split, df, path, meta in todo:
                desc = f'{encoder_kind}/{target_name}/{split}'
                if needs_dist(encoder_kind):
                    dataset = PBPKDistDataset(df, target_name, dist_store)
                    feat, animal, value = extract_dist_features(
                        encoder, dataset, tokenizer, device, args.max_len, desc=desc)
                else:
                    dataset = SMILESDataset_single_PBPK(df, target_name)
                    feat, animal, value = extract_features(
                        encoder, dataset, tokenizer, device, args.extract_batch_size,
                        args.max_len, desc=desc)
                assert len(feat) == len(df), f'{len(feat)} extracted vs {len(df)} rows'
                save_cache(path, feat, animal, value, meta)
                print(f'[save] {tuple(feat.shape)}  mean|z|={np.abs(feat).mean():.3f}  -> {path}')
                n_written += 1
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f'\n{n_written} cache file(s) written, {n_skipped} already up to date -> {args.cache_dir}')


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
        except Exception as e:                       # noqa: BLE001 -- listing must not die on one bad file
            print(f'{f}\n    [unreadable] {e}')
    print(f'\n{len(files)} file(s), {total/1e6:.1f} MB in {cache_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder', nargs='+', default=['fusion', 'text'], choices=list(ENCODERS),
                        help="fusion = new_..._v1's embedding, text = new_..._v2/v3's (shared)")
    parser.add_argument('--checkpoints', nargs='+',
                        default=['./Pretrain/trimodal_pubchem100m_96_step=173008.ckpt',
                                 './Pretrain/trimodal_pubchem100m_96_step=216260.ckpt'],
                        help='One cache file per checkpoint -- ensemble members never share.')
    parser.add_argument('--input', default='src-iwata-pk', type=str)
    parser.add_argument('--target_name', nargs='+', default=['human_CL', 'human_VDss', 'human_fup'])
    parser.add_argument('--splits', nargs='+', default=['train', 'valid'], choices=list(SPLITS))
    parser.add_argument('--pool', action='store_true',
                        help="ignore the Train/Valid columns and cache one split-independent "
                             "'all' file per (encoder, checkpoint, input, target). Animal values "
                             'are stored UNSCALED so k-fold can refit the scaler per fold.')
    parser.add_argument('--imputer', action='store_true',
                        help='passed through to cv_data_transform (which currently ignores it), '
                             'so it is NOT part of the cache key')
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    parser.add_argument('--max_len', default=100, type=int, help='SMILES token truncation length')
    parser.add_argument('--extract_batch_size', default=64, type=int)
    parser.add_argument('--bert_config_text', default=DEFAULT_BERT_CONFIG_TEXT)
    parser.add_argument('--bert_config_property', default=DEFAULT_BERT_CONFIG_PROPERTY)
    parser.add_argument('--bert_config_dist', default=DEFAULT_BERT_CONFIG_DIST)
    # DIST_CUTOFF / DIST_MAX_PAIR / DIST_TRUNCATE are module constants, not flags:
    # they are part of the forward path, and a flag could silently write two
    # different conventions into one cache filename. Edit the constants instead --
    # they are in the encoder `spec`, so old caches invalidate themselves.
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
