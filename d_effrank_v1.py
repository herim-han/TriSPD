"""
d_effrank_v1.py
===============
Effective-rank analysis of the trimodal SPMM contrastive embedding space, to test
the hypothesis that the added 3D-local-graph (`dist`) modality is an AUXILIARY
signal: it shares most of its content with `text`, injects a small amount of
local-explicit information `text` cannot cover, and does NOT replace a modality.

DeCUR connection
----------------
DeCUR (2309.05300) diagnoses *dimensional collapse* via the cross-correlation
matrix and its redundancy (off-diagonal) structure. SPMM is trained with InfoNCE
(ALBEF/MoCo), NOT Barlow-Twins, so its embedding dimensions are NOT axis-aligned
-- DeCUR's per-dimension diagonal histogram (Eq.7) is therefore NOT directly
valid here. We use the *rotation-invariant* generalization instead: the singular
spectrum of each modality's centered embedding matrix -> effective rank.

What we measure (all on the 256-d L2-normalized contrastive feats `*_feat`,
i.e. the exact space where the InfoNCE loss lives; CLS-768 reported as a check):
  - per-modality effective rank   (P1: dist learned & not collapsed; P2: dist<text)
  - joint effective rank & increments (P3/P4: dist adds a small UNIQUE subspace
                                       on top of text / on top of prop+text)

No retraining. We replicate `extract_feature` (trimodal_bert_models_v3) WITHOUT
the 50% MPM/MDM masking, to read the clean learned representation.

Usage
-----
python d_effrank_v1.py \
    --checkpoint ./Pretrain/trimodal_pubchem100m_96_step=216260.ckpt \
    --val_lmdb   ./valid_set.lmdb \
    --num_samples 4096 --batch_size 64 \
    --outdir ./probe_results

Outputs (under --outdir, tagged by checkpoint stem)
  - effrank_<tag>_metrics.json   per-modality + joint effective ranks
  - effrank_<tag>_spectra.npz    singular-value spectra (for plotting)
  - effrank_<tag>_spectrum.png   normalized cumulative spectra (if matplotlib)
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import ValLMDBDataset, tri_collate_fn
from trimodal_bert_models_v3 import SPMM


# ----------------------------------------------------------------------------- #
# effective-rank metrics (all on a centered N x K matrix; rotation-invariant)
# ----------------------------------------------------------------------------- #
def _singular_values(X, standardize=False, eps=1e-12):
    """X: (N, K) float64. Center (and optionally z-score each dim) -> singular values."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if standardize:
        sd = X.std(axis=0, keepdims=True)
        X = X / np.maximum(sd, eps)
    # singular values of centered data (sqrt of covariance eigenvalues * (N-1))
    s = np.linalg.svd(X, compute_uv=False)
    return s


def effective_rank_metrics(X, standardize=False):
    """
    Returns a dict of rotation-invariant rank summaries.
      eff_rank   : Roy-Vetterli entropy rank  exp(-sum p_k log p_k),  p_k = s_k / sum s
      part_ratio : participation ratio        (sum s^2)^2 / sum s^4   (== (sum lam)^2/sum lam^2)
      stable_rank: ||X||_F^2 / ||X||_2^2       = sum s^2 / s_max^2
      dim99/dim95: # components for 99%/95% of spectral energy (sum s^2)
      K          : ambient dimension
    """
    s = _singular_values(X, standardize=standardize)
    s = s[s > 0]
    K = int(np.asarray(X).shape[1])
    if s.size == 0:
        return dict(eff_rank=0.0, part_ratio=0.0, stable_rank=0.0,
                    dim99=0, dim95=0, K=K, s_max=0.0)
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))  #Roy-Vetterli entropy rank = exp(H)
    lam = s ** 2
    part_ratio = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))
    stable_rank = float(lam.sum() / (lam.max() + 1e-12))
    energy = np.cumsum(lam) / lam.sum()
    dim99 = int(np.searchsorted(energy, 0.99) + 1)
    dim95 = int(np.searchsorted(energy, 0.95) + 1)
    return dict(eff_rank=eff_rank, part_ratio=part_ratio, stable_rank=stable_rank,
                dim99=dim99, dim95=dim95, K=K, s_max=float(s.max()))


def joint_effrank(blocks, standardize=True):
    """blocks: list of (N, K_i) arrays (same N). z-score each block so neither
    dominates by scale, concatenate, then effective rank of the union space."""
    Z = []
    for B in blocks:
        B = np.asarray(B, dtype=np.float64)
        B = B - B.mean(axis=0, keepdims=True)
        if standardize:
            B = B / np.maximum(B.std(axis=0, keepdims=True), 1e-12)
        Z.append(B)
    Z = np.concatenate(Z, axis=1)
    return effective_rank_metrics(Z, standardize=False)  # already standardized per-block


# ----------------------------------------------------------------------------- #
# feature extraction (clean = no MPM/MDM masking), mirrors extract_feature
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def bi_extract_all_feats(model, tokenizer, loader, device, num_samples):
    feats = {k: [] for k in ['prop_feat', 'text_feat',
                             'prop_cls', 'text_cls']}
    n = 0
    for prop, smiles, atom_pair, dist in loader:
        prop = prop.to(device)
        atom_pair = atom_pair.to(device)
        dist = dist.to(device)

        text = list(smiles)
        tin = tokenizer(text, padding='longest', truncation=True, max_length=100,
                        return_tensors='pt').to(device)
        ids = tin.input_ids[:, 1:]       #text_ids
        mask = tin.attention_mask[:, 1:] #text_attention_mask

        # --- text -------------------------------------------------------------
        t_embeds = model.text_encoder.bert(ids, attention_mask=mask,
                                           return_dict=True,
                                           mode='text').last_hidden_state
        t_cls = t_embeds[:, 0, :]
        t_feat = F.normalize(model.text_proj(t_cls), dim=-1)

        # --- prop (NO masking) ------------------------------------------------
        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        p_embeds = model.property_encoder(inputs_embeds=properties,
                                          return_dict=True).last_hidden_state
        p_cls = p_embeds[:, 0, :]
        p_feat = F.normalize(model.property_proj(p_cls), dim=-1)


        feats['prop_feat'].append(p_feat.float().cpu().numpy())
        feats['text_feat'].append(t_feat.float().cpu().numpy())
        feats['prop_cls'].append(p_cls.float().cpu().numpy())
        feats['text_cls'].append(t_cls.float().cpu().numpy())

        n += prop.size(0)
        print(f'  collected {n}/{num_samples}', end='\r')
        if n >= num_samples:
            break
    print()
    return {k: np.concatenate(v, axis=0)[:num_samples] for k, v in feats.items()}

@torch.no_grad()
def extract_all_feats(model, tokenizer, loader, device, num_samples):
    feats = {k: [] for k in ['prop_feat', 'text_feat', 'dist_feat',
                             'prop_cls', 'text_cls', 'dist_cls']}
    n = 0
    for prop, smiles, atom_pair, dist in loader:
        prop = prop.to(device)
        atom_pair = atom_pair.to(device)
        dist = dist.to(device)

        text = list(smiles)
        tin = tokenizer(text, padding='longest', truncation=True, max_length=100,
                        return_tensors='pt').to(device)
        ids = tin.input_ids[:, 1:]       #text_ids
        mask = tin.attention_mask[:, 1:] #text_attention_mask

        # --- text -------------------------------------------------------------
        t_embeds = model.text_encoder.bert(ids, attention_mask=mask,
                                           return_dict=True,
                                           mode='text').last_hidden_state
        t_cls = t_embeds[:, 0, :]
        t_feat = F.normalize(model.text_proj(t_cls), dim=-1)

        # --- prop (NO masking) ------------------------------------------------
        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        p_embeds = model.property_encoder(inputs_embeds=properties,
                                          return_dict=True).last_hidden_state
        p_cls = p_embeds[:, 0, :]
        p_feat = F.normalize(model.property_proj(p_cls), dim=-1)

        # --- dist (NO masking) ------------------------------------------------
        df = model.dist_embed_layer(atom_pair, dist)
        distances = torch.cat(
            [model.dist_cls.expand(df.size(0), -1, -1), df], dim=1)
        d_embeds = model.dist_encoder(inputs_embeds=distances,
                                      return_dict=True).last_hidden_state
        d_cls = d_embeds[:, 0, :]
        d_feat = F.normalize(model.dist_proj(d_cls), dim=-1)

        feats['prop_feat'].append(p_feat.float().cpu().numpy())
        feats['text_feat'].append(t_feat.float().cpu().numpy())
        feats['dist_feat'].append(d_feat.float().cpu().numpy())
        feats['prop_cls'].append(p_cls.float().cpu().numpy())
        feats['text_cls'].append(t_cls.float().cpu().numpy())
        feats['dist_cls'].append(d_cls.float().cpu().numpy())

        n += prop.size(0)
        print(f'  collected {n}/{num_samples}', end='\r')
        if n >= num_samples:
            break
    print()
    return {k: np.concatenate(v, axis=0)[:num_samples] for k, v in feats.items()}


def build_config():
    return {
        'property_width': 768, 'dist_width': 768, 'embed_dim': 256,
        'batch_size': 64, 'temp': 0.07, 'mlm_probability': 0.15,
        'queue_size': 24960, 'momentum': 0.995, 'alpha': 0.4,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'bert_config_dist': './config_bert_dist.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 10, 'min_lr': 5e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20,
                      'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='./Pretrain/trimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--val_lmdb', default='./valid_set.lmdb')
    ap.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    ap.add_argument('--num_samples', type=int, default=8192)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--outdir', default='./probe_results')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_shuffle', action='store_true',
                    help='take the first N contiguous keys instead of a random subset')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    device = torch.device(args.device)

    # ---- tokenizer (mirror pretrain) ----------------------------------------
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False,
                              do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab, unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250)

    # ---- model + checkpoint -------------------------------------------------
    print(f'building trimodal SPMM, loading {args.checkpoint}')
    model = SPMM(config=build_config(), tokenizer=tokenizer, no_train=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    for key in list(state.keys()):          # legacy _unk -> _mask (as in d_regression_*)
        if '_unk' in key:
            state[key.replace('_unk', '_mask')] = state.pop(key)
    msg = model.load_state_dict(state, strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith(('property_encoder_m',
            'text_encoder_m', 'dist_encoder_m', 'property_proj_m', 'text_proj_m',
            'dist_proj_m')) and 'queue' not in k]
    print(f'  missing(non-momentum/queue): {len(miss)}  unexpected: {len(msg.unexpected_keys)}')
    if miss[:10]:
        print('  e.g.', miss[:10])
    model.eval().to(device)

    # ---- data ---------------------------------------------------------------
    ds = ValLMDBDataset(args.val_lmdb)
    if args.no_shuffle:
        sub = ds
        print(f'using first {args.num_samples} contiguous keys of {len(ds)}')
    else:
        # representative random subset of the ~1.65M validation molecules
        idx = np.random.permutation(len(ds))[:args.num_samples]
        sub = Subset(ds, idx.tolist())
        print(f'using random subset {args.num_samples} / {len(ds)} (seed={args.seed})')
    loader = DataLoader(sub, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, collate_fn=tri_collate_fn)
    print(f'extracting feats over up to {args.num_samples} molecules ...')
    feats = extract_all_feats(model, tokenizer, loader, device, args.num_samples)
    N = feats['prop_feat'].shape[0]
    print(f'collected N={N}')

    # ---- per-modality effective rank ----------------------------------------
    results = {'checkpoint': args.checkpoint, 'N': N, 'per_modality': {}, 'joint': {}}
    print('\n=== per-modality effective rank (256-d contrastive feat) ===')
    print(f'{"modality":10s} {"eff_rank":>9s} {"part_ratio":>11s} {"stable_rank":>12s} '
          f'{"dim95":>6s} {"dim99":>6s} {"K":>4s}')
    for m in ['text_feat', 'prop_feat', 'dist_feat']:
        r = effective_rank_metrics(feats[m])                 # covariance-based
        rc = effective_rank_metrics(feats[m], standardize=True)  # correlation-based
        results['per_modality'][m] = {'cov': r, 'corr': rc}
        print(f'{m:10s} {r["eff_rank"]:9.2f} {r["part_ratio"]:11.2f} '
              f'{r["stable_rank"]:12.2f} {r["dim95"]:6d} {r["dim99"]:6d} {r["K"]:4d}')
    print('  (768-d CLS check)')
    for m in ['text_cls', 'prop_cls', 'dist_cls']:
        r = effective_rank_metrics(feats[m])
        results['per_modality'][m] = {'cov': r}
        print(f'{m:10s} {r["eff_rank"]:9.2f} {r["part_ratio"]:11.2f} '
              f'{r["stable_rank"]:12.2f} {r["dim95"]:6d} {r["dim99"]:6d} {r["K"]:4d}')

    # ---- joint effective rank & increments (P3 / P4) ------------------------
    print('\n=== joint effective rank & dist increment (z-scored blocks) ===')
    combos = {
        'text': ['text_feat'],
        'prop': ['prop_feat'],
        'dist': ['dist_feat'],
        'text+dist': ['text_feat', 'dist_feat'],
        'prop+text': ['prop_feat', 'text_feat'],
        'prop+text+dist': ['prop_feat', 'text_feat', 'dist_feat'],
    }
    jr = {}
    for name, keys in combos.items():
        r = joint_effrank([feats[k] for k in keys])
        jr[name] = r
        results['joint'][name] = r
        print(f'{name:16s} eff_rank={r["eff_rank"]:7.2f}  part_ratio={r["part_ratio"]:7.2f}  '
              f'dim99={r["dim99"]:4d}  K={r["K"]:4d}')
    d_over_text = jr['text+dist']['eff_rank'] - jr['text']['eff_rank']
    d_over_pt = jr['prop+text+dist']['eff_rank'] - jr['prop+text']['eff_rank']
    results['increments'] = {
        'eff_rank_text': jr['text']['eff_rank'],
        'eff_rank_dist': jr['dist']['eff_rank'],
        'delta_dist_over_text': d_over_text,                # P3
        'delta_dist_over_proptext': d_over_pt,              # P4
        'sum_text_dist': jr['text']['eff_rank'] + jr['dist']['eff_rank'],
        'joint_text_dist': jr['text+dist']['eff_rank'],
    }
    print(f'\n  P2  dist eff_rank ({jr["dist"]["eff_rank"]:.2f}) vs text ({jr["text"]["eff_rank"]:.2f})'
          f'  -> {"dist<text OK" if jr["dist"]["eff_rank"] < jr["text"]["eff_rank"] else "dist>=text (peer-like)"}')
    print(f'  P3  r(text+dist) - r(text) = {d_over_text:+.2f}   '
          f'(small +ve = auxiliary unique subspace; ~0 = redundant; large = independent)')
    print(f'  P4  r(prop+text+dist) - r(prop+text) = {d_over_pt:+.2f}')
    print(f'      sum r(text)+r(dist)={results["increments"]["sum_text_dist"]:.2f} vs '
          f'joint r(text+dist)={jr["text+dist"]["eff_rank"]:.2f}  '
          f'(joint<<sum => heavy sharing)')

    # ---- save ---------------------------------------------------------------
    with open(f'{args.outdir}/effrank_{tag}_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    spectra = {m: _singular_values(feats[m]) for m in
               ['text_feat', 'prop_feat', 'dist_feat']}
    np.savez(f'{args.outdir}/effrank_{tag}_spectra.npz', **spectra)
    print(f'\nsaved metrics -> {args.outdir}/effrank_{tag}_metrics.json')

    # ---- optional plot ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for m, c in [('text_feat', 'C0'), ('prop_feat', 'C1'), ('dist_feat', 'C2')]:
            s = spectra[m]
            lam = s ** 2
            plt.plot(np.cumsum(lam) / lam.sum(), c, label=m)
        plt.axhline(0.99, ls='--', c='gray', lw=0.8)
        plt.xlabel('# singular components')
        plt.ylabel('cumulative spectral energy')
        plt.title(f'effective rank spectrum\n{tag}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{args.outdir}/effrank_{tag}_spectrum.png', dpi=140)
        print(f'saved plot   -> {args.outdir}/effrank_{tag}_spectrum.png')
    except Exception as e:
        print(f'(plot skipped: {e})')


if __name__ == '__main__':
    main()
