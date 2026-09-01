"""
d_cka_v1.py
===========
Quantify how much the 3D-local-graph (`dist`) modality actually SHARES with `text`
(and prop), using three rotation-invariant measures on the 256-d contrastive feats:

  1. CKA (linear + RBF)        -- normalized representational similarity in [0,1].
                                  CKA(text,dist) near 1 => text & dist encode the
                                  same thing; near 0 => independent.
  2. CCA canonical correlations -- the shared-subspace spectrum. #(rho>tau) = number
                                  of shared directions; their magnitude = how aligned.
  3. Cross-prediction R^2       -- fraction of one modality's variance LINEARLY
                                  reconstructable from another. R^2(dist<-text) is the
                                  most direct "what fraction of dist is shared with
                                  text" number; 1-R^2 = dist-unique fraction.

Why this and not DeCUR's Eq.7 diagonal: SPMM is InfoNCE (whole-vector cosine),
so embedding axes are NOT aligned across modalities -- per-dimension diagonal
correlation is meaningless. CKA/CCA/cross-pred are all basis-independent, so they
read the shared content regardless of how each modality rotated its own basis.
This closes the caveat from d_effrank_v1 (linear rank-overlap is only a LOWER
bound on sharing; CKA/CCA give the proper, rotation-invariant answer).

Reuses extraction (build_config, extract_all_feats) from d_effrank_v1 -- same
single-[CLS] tokenization, same clean (un-masked) feats.

Usage
-----
python d_cka_v1.py \
    --checkpoint ./Pretrain/trimodal_pubchem100m_96_step=216260.ckpt \
    --val_lmdb ./valid_set.lmdb --num_samples 8192 --no_shuffle
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, WordpieceTokenizer
from sklearn.linear_model import Ridge

from dataset import ValLMDBDataset, tri_collate_fn
from trimodal_bert_models_v3 import SPMM
from d_effrank_v1 import build_config, extract_all_feats


# --------------------------------------------------------------------------- #
# similarity / sharing measures (all on z-scored, centered features)
# --------------------------------------------------------------------------- #
def _zscore(X):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    return X / np.maximum(X.std(0, keepdims=True), 1e-12)


def linear_cka(X, Y):
    """Linear CKA in [0,1]. X,Y: (N,d) (will be centered)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    return float((xty ** 2).sum() /
                 (np.sqrt((xtx ** 2).sum()) * np.sqrt((yty ** 2).sum()) + 1e-12))


def _rbf_gram(X, sigma=None):
    sq = np.sum(X ** 2, 1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0)
    if sigma is None:                       # median heuristic
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med / 2) + 1e-12
    return np.exp(-d2 / (2 * sigma ** 2))


def rbf_cka(X, Y, sub=2048, seed=0):
    """RBF CKA on a random subsample (Gram is N^2)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N > sub:
        idx = rng.permutation(N)[:sub]
        X, Y = X[idx], Y[idx]
    K = _rbf_gram(X)
    L = _rbf_gram(Y)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_kl = (Kc * Lc).sum()
    hsic_kk = (Kc * Kc).sum()
    hsic_ll = (Lc * Lc).sum()
    return float(hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-12))


def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(w ** -0.5) @ V.T


def cca_corrs(X, Y, reg=1e-3):
    """Canonical correlations (closed-form, ridge-regularized). Returns rho desc."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    n = X.shape[0]
    Cxx = X.T @ X / n + reg * np.eye(X.shape[1])
    Cyy = Y.T @ Y / n + reg * np.eye(Y.shape[1])
    Cxy = X.T @ Y / n
    M = _inv_sqrt(Cxx) @ Cxy @ _inv_sqrt(Cyy)
    rho = np.linalg.svd(M, compute_uv=False)
    return np.clip(rho, 0.0, 1.0)


def cross_pred_r2(X, Y, alpha=1.0, frac=0.5, seed=0):
    """Aggregate R^2 predicting Y from X (z-scored, ridge, train/test split).
    = fraction of Y's total variance linearly reconstructable from X. 
    #z-scored/ridge-based R2 prediction"""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    ntr = int(N * frac)
    tr, te = perm[:ntr], perm[ntr:]
    # standardize using train stats
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-12)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    muy, sdy = Y[tr].mean(0), np.maximum(Y[tr].std(0), 1e-12)
    Ytr, Yte = (Y[tr] - muy) / sdy, (Y[te] - muy) / sdy
    model = Ridge(alpha=alpha).fit(Xtr, Ytr)
    Yp = model.predict(Xte)
    ss_res = ((Yte - Yp) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


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
    ap.add_argument('--no_shuffle', action='store_true')
    ap.add_argument('--ridge_alpha', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False,
                              do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab, unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250)

    print(f'building trimodal SPMM, loading {args.checkpoint}')
    model = SPMM(config=build_config(), tokenizer=tokenizer, no_train=True)
    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck['state_dict'] if 'state_dict' in ck else ck
    for key in list(state.keys()):
        if '_unk' in key:
            state[key.replace('_unk', '_mask')] = state.pop(key)
    msg = model.load_state_dict(state, strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith((
        'property_encoder_m', 'text_encoder_m', 'dist_encoder_m',
        'property_proj_m', 'text_proj_m', 'dist_proj_m')) and 'queue' not in k]
    print(f'  missing(non-momentum/queue): {len(miss)}  unexpected: {len(msg.unexpected_keys)}')
    model.eval().to(device)

    ds = ValLMDBDataset(args.val_lmdb)
    if args.no_shuffle:
        sub = ds
        print(f'using first {args.num_samples} contiguous keys of {len(ds)}')
    else:
        idx = np.random.permutation(len(ds))[:args.num_samples]
        sub = Subset(ds, idx.tolist())
        print(f'using random subset {args.num_samples}/{len(ds)} (seed={args.seed})')
    loader = DataLoader(sub, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, collate_fn=tri_collate_fn)
    feats = extract_all_feats(model, tokenizer, loader, device, args.num_samples)
    T, D, P = feats['text_feat'], feats['dist_feat'], feats['prop_feat']
    N = T.shape[0]
    print(f'collected N={N}\n')

    pairs = [('text', 'dist', T, D), ('text', 'prop', T, P), ('prop', 'dist', P, D)]
    results = {'checkpoint': args.checkpoint, 'N': N, 'pairs': {}}

    print('=== CKA (representational similarity, 0=independent .. 1=identical) ===')
    print(f'{"pair":12s} {"CKA_linear":>11s} {"CKA_rbf":>9s}')
    for a, b, X, Y in pairs:
        ckal = linear_cka(X, Y)
        ckar = rbf_cka(X, Y, seed=args.seed)
        results['pairs'][f'{a}-{b}'] = {'cka_linear': ckal, 'cka_rbf': ckar}
        print(f'{a+"-"+b:12s} {ckal:11.3f} {ckar:9.3f}')

    print('\n=== CCA canonical correlations (shared-subspace spectrum) ===')
    print(f'{"pair":12s} {"mean_rho":>9s} {"#rho>.7":>8s} {"#rho>.5":>8s} {"#rho>.3":>8s}  top5')
    for a, b, X, Y in pairs:
        rho = cca_corrs(_zscore(X), _zscore(Y))
        r = results['pairs'][f'{a}-{b}']
        r.update(dict(cca_mean=float(rho.mean()),
                      cca_gt07=int((rho > 0.7).sum()),
                      cca_gt05=int((rho > 0.5).sum()),
                      cca_gt03=int((rho > 0.3).sum()),
                      cca_top5=[round(float(x), 3) for x in rho[:5]]))
        print(f'{a+"-"+b:12s} {rho.mean():9.3f} {int((rho>0.7).sum()):8d} '
              f'{int((rho>0.5).sum()):8d} {int((rho>0.3).sum()):8d}  '
              f'{[round(float(x),2) for x in rho[:5]]}')

    print('\n=== cross-prediction R^2 (fraction of TARGET variance explained by SOURCE) ===')
    print(f'{"target<-source":18s} {"R^2":>7s}   (1-R^2 = target-unique fraction)')
    cross = [('dist', 'text', D, T), ('text', 'dist', T, D),
             ('dist', 'prop', D, P), ('prop', 'dist', P, D),
             ('text', 'prop', T, P), ('prop', 'text', P, T)]
    for tgt, src, Y, X in cross:
        r2 = cross_pred_r2(X, Y, alpha=args.ridge_alpha, seed=args.seed)
        results['pairs'].setdefault(f'{src}-{tgt}', {})
        results['pairs'][f'{src}-{tgt}'][f'r2_{tgt}_from_{src}'] = r2
        print(f'{tgt+"<-"+src:18s} {r2:7.3f}   unique={1-r2:6.3f}')

    # headline for the hypothesis
    r2_dt = cross_pred_r2(T, D, alpha=args.ridge_alpha, seed=args.seed)   # dist<-text
    r2_td = cross_pred_r2(D, T, alpha=args.ridge_alpha, seed=args.seed)   # text<-dist
    print('\n--- headline (text vs dist) ---')
    print(f'  CKA_linear(text,dist) = {results["pairs"]["text-dist"]["cka_linear"]:.3f}')
    print(f'  R^2(dist <- text)     = {r2_dt:.3f}   => {100*r2_dt:.0f}% of dist is '
          f'linearly shared with text; {100*(1-r2_dt):.0f}% is dist-unique')
    print(f'  R^2(text <- dist)     = {r2_td:.3f}')
    print(f'  CCA shared dirs(rho>.5) = {results["pairs"]["text-dist"]["cca_gt05"]} / {min(T.shape[1],D.shape[1])}')

    with open(f'{args.outdir}/cka_{tag}_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {args.outdir}/cka_{tag}_metrics.json')


if __name__ == '__main__':
    main()
