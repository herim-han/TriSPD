"""Bimodal vs trimodal comparison on the same molecules: effective rank, CKA,
dist-coverage, and layer-wise cross-model CKA, into one metrics JSON."""
import os
import gc
import json
import argparse
import numpy as np
import torch

from .data import build_tokenizer, make_loader
from .models import load_model
from .extract import extract_all_feats, bi_extract_all_feats, extract_layerwise_cls
from .effrank import _singular_values, effective_rank_metrics
from .metrics import linear_cka, cross_pred_r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bi_ckpt', default='./Pretrain/bi_random/bimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--tri_ckpt', default='./Pretrain/tri_random/trimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--val_lmdb', default='./valid_set.lmdb')
    ap.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    ap.add_argument('--num_samples', type=int, default=8192)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--outdir', default='./')
    ap.add_argument('--outname', default='bivstri_metrics.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_shuffle', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(args.vocab_filename)

    # PART 1: pooled [CLS] global feature -> eff_rank / CKA / dist-coverage
    bm = load_model(args.bi_ckpt, tokenizer, device)
    bf = bi_extract_all_feats(bm, tokenizer,
                           make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                                       args.num_workers, args.no_shuffle, args.seed),
                           device, args.num_samples)
    bi_text, bi_prop = bf['text_feat'], bf['prop_feat']
    del bm; gc.collect(); torch.cuda.empty_cache()

    # same molecules, same permutation seed
    tm = load_model(args.tri_ckpt, tokenizer, device)
    tf = extract_all_feats(tm, tokenizer,
                           make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                                       args.num_workers, args.no_shuffle, args.seed),
                           device, args.num_samples)
    tri_text, tri_prop, tri_dist = tf['text_feat'], tf['prop_feat'], tf['dist_feat']
    del tm; gc.collect(); torch.cuda.empty_cache()

    reps = {'bi_text': bi_text, 'bi_prop': bi_prop,
            'tri_text': tri_text, 'tri_prop': tri_prop, 'tri_dist': tri_dist}
    names = list(reps.keys())
    N = bi_text.shape[0]

    er = {}
    for k in names:
        er[k] = effective_rank_metrics(reps[k])

    spectra = {}
    for k in names:
        s = _singular_values(reps[k]); lam = s ** 2
        spectra[k] = (np.cumsum(lam) / lam.sum()).tolist()

    M = np.zeros((5, 5))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = linear_cka(reps[a], reps[b])
    resh3_text = M[names.index('bi_text'), names.index('tri_text')]
    resh4_prop = M[names.index('bi_prop'), names.index('tri_prop')]

    cov = {}
    for label, src in [('dist<-text', tri_text), ('dist<-prop', tri_prop),
                       ('dist<-text+prop', np.concatenate([tri_text, tri_prop], 1))]:
        cov[label] = cross_pred_r2(src, tri_dist, seed=args.seed)

    # prop<-text is the control: measured in both models, so the gap is
    # attributable to the third modality
    prop_cov = {}
    for label, src, tgt in [('prop<-text (tri)', tri_text, tri_prop),
                            ('prop<-text (bi)', bi_text, bi_prop)]:
        prop_cov[label] = cross_pred_r2(src, tgt, seed=args.seed)
    d_prop = prop_cov['prop<-text (tri)'] - prop_cov['prop<-text (bi)']

    results = {'N': N, 'eff_rank': er,
               'cka_matrix': {names[i]: {names[j]: float(M[i, j]) for j in range(5)} for i in range(5)},
               'reshaping': {'bi_text~tri_text': float(resh3_text),
                             'bi_prop~tri_prop': float(resh4_prop)},
               'dist_coverage_r2': cov,
               'prop_coverage_r2': prop_cov,
               'spectra_cumenergy': spectra}

    # PART 2: layer-wise [CLS] cross-model CKA
    bm = load_model(args.bi_ckpt, tokenizer, device)
    bf_l = extract_layerwise_cls(
        bm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del bm; gc.collect(); torch.cuda.empty_cache()

    tm = load_model(args.tri_ckpt, tokenizer, device)
    tf_l = extract_layerwise_cls(
        tm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del tm; gc.collect(); torch.cuda.empty_cache()

    L_text = len(bf_l['text_layers'])
    L_prop = len(bf_l['prop_layers'])
    assert L_text == len(tf_l['text_layers']), \
        f"text layer count mismatch: bi={L_text} vs tri={len(tf_l['text_layers'])}"
    assert L_prop == len(tf_l['prop_layers']), \
        f"prop layer count mismatch: bi={L_prop} vs tri={len(tf_l['prop_layers'])}"

    cka_text = [linear_cka(bf_l['text_layers'][k], tf_l['text_layers'][k])
                for k in range(L_text)]
    cka_prop = [linear_cka(bf_l['prop_layers'][k], tf_l['prop_layers'][k])
                for k in range(L_prop)]
    cka_text_proj = linear_cka(bf_l['text_feat'], tf_l['text_feat'])
    cka_prop_proj = linear_cka(bf_l['prop_feat'], tf_l['prop_feat'])

    results.update({
        'L_text_hidden_states': L_text,
        'L_prop_hidden_states': L_prop,
        'cka_text_per_layer': [float(x) for x in cka_text],
        'cka_prop_per_layer': [float(x) for x in cka_prop],
        'cka_text_post_proj': float(cka_text_proj),
        'cka_prop_post_proj': float(cka_prop_proj),
    })

    with open(f'{args.outdir}/{args.outname}', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'N={N}  dist_unique={1 - cov["dist<-text+prop"]:.3f}  '
          f'prop_coupling_delta(tri-bi)={d_prop:+.3f}')
    print(f'saved metrics -> {args.outdir}/{args.outname}')

if __name__ == '__main__':
    main()
