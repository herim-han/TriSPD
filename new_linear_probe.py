"""Linear probe over the cached frozen embeddings: how much PK signal is linearly
readable from each representation?

Compares the feature variants sitting in feat_cache/ :

    fusion_pe   fusion__*.npz      text[l0-5] x property_encoder(prop) -> fusion[l6-11] CLS
                                   (the pretraining path, new_extract_features.py)
    fusion_raw  bk_fusion__*.npz   same but property_encoder SKIPPED -- raw property_embed
                                   as K/V, i.e. what d_regression_frozen_v1 has always used
    text        text__*.npz        text[l0-5] CLS, SMILES only (v2/v3)

A probe is ridge regression on standardised features, alpha chosen by RidgeCV
inside each fold, so nothing about the held-out rows leaks into the fit. Two
numbers per row:

    cv_R2     5-fold out-of-fold R2 on the TRAIN split (n~700) -- the stable one
    valid_R2  fit on all of train, score the Valid split (n~46) -- same split the
              training scripts report on, but small enough to be noisy

`animal_only` is a reference row, not a representation: the 9 animal PK columns
(median-imputed + missing indicators), no embedding at all. It says what the
target looks like without any molecular representation.

    python new_linear_probe.py                                   # CL + VDss, 4 ckpts
    python new_linear_probe.py --target_name human_CL --variants fusion_pe text
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import new_extract_features as nef

# variant -> candidate (encoder kind for the filename, path prefix under --cache_dir);
# the first candidate that exists wins. fusion_raw is now a real encoder kind, but the
# pre-change `fusion` caches that were backed up as bk_fusion__*.npz are the same thing,
# so they stay as a fallback for datasets that were never re-extracted.
VARIANTS = {
    'fusion_pe':  [('fusion', '')],
    'fusion_raw': [('fusion_raw', ''), ('fusion', 'bk/bk_'), ('fusion', 'bk_')],
    'text':       [('text', '')],
}

ALPHAS = np.logspace(-2, 6, 25)


def load_variant(variant, ckpt, args, target_name, split):
    for encoder, prefix in VARIANTS[variant]:
        fn = nef.cache_filename(encoder, ckpt, args.input, target_name, split,
                                args.max_len, args.vocab_filename)
        path = os.path.join(args.cache_dir, prefix + fn)
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                return z['feat'], z['animal'], z['value']
    return None


def make_probe():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def run_probe(Xtr, ytr, Xva, yva, seed, n_folds, n_repeats):
    """(cv_R2 mean, cv_R2 std, train_R2, valid_R2, alpha, [oof per repeat], valid_pred).

    cross_val_predict refits the whole pipeline -- scaler AND RidgeCV alpha
    search -- inside every fold, so the out-of-fold R2 is not optimistic. The
    fold split is redrawn n_repeats times: with ~0.02 separating some variants,
    a single 5-fold draw cannot tell a real gap from split noise.
    """
    oofs = [cross_val_predict(make_probe(), Xtr, ytr,
                              cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed + r))
            for r in range(n_repeats)]
    cv_r2 = [r2_score(ytr, o) for o in oofs]
    probe = make_probe().fit(Xtr, ytr)
    va = probe.predict(Xva)
    return (float(np.mean(cv_r2)), float(np.std(cv_r2)), r2_score(ytr, probe.predict(Xtr)),
            r2_score(yva, va), float(probe[-1].alpha_), oofs, va)


def boot_idx(n, n_boot, seed):
    """One resampling matrix reused by every row, so differences can be bootstrapped
    PAIRED: variant A and variant B are scored on the identical resampled molecules."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_r2(y, pred, idx):
    return np.array([r2_score(y[b], pred[b]) for b in idx])


def ci(vals, lo=2.5, hi=97.5):
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def animal_matrix(animal):
    """9 animal PK columns -> median-imputed values + missing indicators."""
    return np.hstack([animal, np.isnan(animal).astype(np.float32)])


def family(ckpt):
    """Ensemble members are grouped by pretraining family, as ckpt_prefix() does."""
    if 'trimodal' in ckpt:
        return 'tri'
    if 'bimodal' in ckpt:
        return 'bi'
    return 'other'


def main(args):
    rows = []
    preds = {}          # (target, variant, ckpt label) -> (y_valid, valid_pred)
    for target_name in args.target_name:
        try:
            splits = nef.load_splits(args.input, target_name, args.imputer)
        except KeyError as e:
            print(f'[skip] {target_name}: {args.input}.csv has no {e} column')
            continue

        # ---- reference row: animal PK columns only, no embedding ----
        ref = load_variant(args.variants[0], args.checkpoints[0], args, target_name, 'train')
        if ref is None:
            print(f'[skip] {target_name}: no cache for {args.variants[0]}')
            continue
        _, a_tr, y_tr = ref
        _, a_va, y_va = load_variant(args.variants[0], args.checkpoints[0], args, target_name, 'valid')
        imp = SimpleImputer(strategy='median').fit(animal_matrix(a_tr))
        cv_r2, cv_sd, tr_r2, va_r2, alpha, _, vp = run_probe(
            imp.transform(animal_matrix(a_tr)), y_tr,
            imp.transform(animal_matrix(a_va)), y_va, args.seed, args.n_folds, args.n_repeats)
        preds[(target_name, 'animal_only', '-')] = (y_va, vp)
        rows.append(dict(target=target_name, variant='animal_only', ckpt='-', n_dim=a_tr.shape[1] * 2,
                         n_train=len(y_tr), n_valid=len(y_va),
                         cv_R2=cv_r2, cv_SD=cv_sd, train_R2=tr_r2, valid_R2=va_r2, alpha=alpha))

        # ---- one probe per (variant, checkpoint), plus a per-family ensemble ----
        for variant in args.variants:
            ens = {}
            for ckpt in args.checkpoints:
                tr = load_variant(variant, ckpt, args, target_name, 'train')
                va = load_variant(variant, ckpt, args, target_name, 'valid')
                if tr is None or va is None:
                    print(f'[skip] {variant} / {os.path.basename(ckpt)} / {target_name}: cache missing')
                    continue
                Xtr, _, ytr = tr
                Xva, _, yva = va
                cv_r2, cv_sd, tr_r2, va_r2, alpha, oofs, vp = run_probe(
                    Xtr, ytr, Xva, yva, args.seed, args.n_folds, args.n_repeats)
                label = nef._stem(ckpt).replace('_pubchem100m_96', '')
                preds[(target_name, variant, label)] = (yva, vp)
                rows.append(dict(target=target_name, variant=variant, ckpt=label,
                                 n_dim=Xtr.shape[1], n_train=len(ytr), n_valid=len(yva),
                                 cv_R2=cv_r2, cv_SD=cv_sd, train_R2=tr_r2, valid_R2=va_r2, alpha=alpha))
                ens.setdefault(family(ckpt), []).append((oofs, ytr, vp, yva))

            # prediction-averaging ensemble, mirroring how the training scripts combine members
            for fam, members in ens.items():
                if len(members) < 2:
                    continue
                # average within a repeat (same fold split for every member), then across repeats
                per_repeat = [np.mean([m[0][r] for m in members], axis=0)
                              for r in range(args.n_repeats)]
                ytr, yva = members[0][1], members[0][3]
                cv_r2 = [r2_score(ytr, o) for o in per_repeat]
                vp = np.mean([m[2] for m in members], axis=0)
                label = f'{fam}-ens({len(members)})'
                preds[(target_name, variant, label)] = (yva, vp)
                rows.append(dict(target=target_name, variant=variant, ckpt=label,
                                 n_dim=np.nan, n_train=len(ytr), n_valid=len(yva),
                                 cv_R2=float(np.mean(cv_r2)), cv_SD=float(np.std(cv_r2)),
                                 train_R2=np.nan, valid_R2=r2_score(yva, vp), alpha=np.nan))

    df = pd.DataFrame(rows)
    if df.empty:
        print('nothing probed')
        return

    # ---- bootstrap the benchmark valid split: CI per row, and PAIRED CIs for
    # every difference. The valid split is small, so a bare point estimate says
    # nothing about whether a 0.02 gap is real; resampling the same molecules for
    # both members of a comparison removes the shared split noise.
    idx_per_target = {t: boot_idx(int(sub.n_valid.iloc[0]), args.n_boot, args.seed)
                      for t, sub in df.groupby('target', sort=False)}
    boots = {k: boot_r2(y, p, idx_per_target[k[0]]) for k, (y, p) in preds.items()}
    los, his = [], []
    for _, r in df.iterrows():
        b = boots[(r.target, r.variant, r.ckpt)]
        lo, hi = ci(b)
        los.append(lo)
        his.append(hi)
    df['valid_lo'], df['valid_hi'] = los, his

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'linear_probe_{args.input}.csv')
    df.to_csv(out, index=False)

    pd.set_option('display.width', 220, 'display.max_rows', 300)
    for target_name, sub in df.groupby('target', sort=False):
        print(f'\n===== {target_name}  (log10 space; n_train={sub.n_train.iloc[0]}, '
              f'n_valid={sub.n_valid.iloc[0]}) =====')
        show = sub.copy()
        show['valid_R2 [95% CI]'] = [f'{r.valid_R2:6.3f} [{r.valid_lo:6.3f},{r.valid_hi:6.3f}]'
                                     for r in sub.itertuples()]
        print(show[['variant', 'ckpt', 'valid_R2 [95% CI]', 'cv_R2', 'cv_SD', 'train_R2', 'alpha']]
              .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    # ---- paired differences on the valid split ----
    def paired(a, b):
        d = boots[a] - boots[b]
        lo, hi = ci(d)
        # share of resamples where a beats b: 0.5 = coin flip, ~1.0 = consistent
        return d.mean(), lo, hi, float((d > 0).mean())

    print(f'\n===== paired bootstrap on the valid split ({args.n_boot} resamples) =====')
    print('diff = A - B on the SAME resampled molecules; win% = share of resamples with A > B\n')
    lines = []
    for target_name in df.target.unique():
        labels = sorted({k[2] for k in preds if k[0] == target_name and 'ens' in k[2]})
        variants = [v for v in args.variants]
        # variant vs variant, within a checkpoint family
        for lab in labels:
            for i, va in enumerate(variants):
                for vb in variants[i + 1:]:
                    ka, kb = (target_name, va, lab), (target_name, vb, lab)
                    if ka in boots and kb in boots:
                        m, lo, hi, w = paired(ka, kb)
                        lines.append((target_name, lab, f'{va} - {vb}', m, lo, hi, w))
        # family vs family, within a variant
        for v in variants:
            tri = [l for l in labels if l.startswith('tri')]
            bi = [l for l in labels if l.startswith('bi')]
            if tri and bi and (target_name, v, tri[0]) in boots and (target_name, v, bi[0]) in boots:
                m, lo, hi, w = paired((target_name, v, tri[0]), (target_name, v, bi[0]))
                lines.append((target_name, v, 'trimodal - bimodal', m, lo, hi, w))
    pdf = pd.DataFrame(lines, columns=['target', 'where', 'comparison', 'diff', 'lo', 'hi', 'win%'])
    pdf['95% CI'] = [f'[{r.lo:6.3f},{r.hi:6.3f}]' for r in pdf.itertuples()]
    pdf['sig'] = np.where((pdf.lo > 0) | (pdf.hi < 0), '*', '')
    print(pdf[['target', 'where', 'comparison', 'diff', '95% CI', 'win%', 'sig']]
          .to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
    pdf.to_csv(os.path.join(args.out_dir, f'linear_probe_{args.input}_paired.csv'), index=False)
    print(f'\n-> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--checkpoints', nargs='+',
                        default=['Pretrain/trimodal_pubchem100m_96_step=173008.ckpt',
                                 'Pretrain/trimodal_pubchem100m_96_step=216260.ckpt',
                                 'Pretrain/bimodal_pubchem100m_96_step=173008.ckpt',
                                 'Pretrain/bimodal_pubchem100m_96_step=216260.ckpt'])
    parser.add_argument('--target_name', nargs='+', default=['human_CL', 'human_VDss', 'human_fup'])
    parser.add_argument('--input', default='src-iwata-pk', type=str)
    parser.add_argument('--imputer', action='store_true')
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--cache_dir', default=nef.DEFAULT_CACHE_DIR, type=str)
    parser.add_argument('--out_dir', default='./probe_results', type=str)
    parser.add_argument('--n_folds', default=5, type=int)
    parser.add_argument('--n_repeats', default=5, type=int,
                        help='how many times the K-fold split is redrawn; cv_SD is the spread over these')
    parser.add_argument('--n_boot', default=2000, type=int,
                        help='bootstrap resamples of the valid split for the CIs and paired tests')
    parser.add_argument('--seed', default=0, type=int)
    main(parser.parse_args())
