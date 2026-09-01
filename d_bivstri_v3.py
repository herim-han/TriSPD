import os
import gc
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import ValLMDBDataset, tri_collate_fn
from d_effrank_v1 import build_config, extract_all_feats, effective_rank_metrics, _singular_values, bi_extract_all_feats
from d_cka_v1 import linear_cka, cross_pred_r2
from d_bivstri_layerwise import extract_layerwise_cls
from d_bivstri_fusion_v1 import extract_fusion_cls


def load_model(ckpt_path, tokenizer, device):
    if 'bimodal' in os.path.basename(ckpt_path): # > Bimodal
        from SPMM_models import SPMM
    else:                                        # > Trimodal
        from trimodal_bert_models_v3 import SPMM
    model = SPMM(config=build_config(), tokenizer=tokenizer, no_train=True)
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck['state_dict'] if 'state_dict' in ck else ck
    for key in list(state.keys()):
        if '_unk' in key:
            state[key.replace('_unk', '_mask')] = state.pop(key)
    msg = model.load_state_dict(state, strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith((
        'property_encoder_m', 'text_encoder_m', 'dist_encoder_m',
        'property_proj_m', 'text_proj_m', 'dist_proj_m')) and 'queue' not in k]
    print(f'  loaded {os.path.basename(ckpt_path)}: '
          f'missing(non-mom/queue)={len(miss)} unexpected={len(msg.unexpected_keys)}')
    return model.eval().to(device)

def make_loader(val_lmdb, num_samples, batch_size, num_workers, no_shuffle, seed):
    ds = ValLMDBDataset(val_lmdb)
    if no_shuffle:
        print('shuffle False')
        sub = ds
    else:
        print('shuffle True')
        idx = np.random.default_rng(seed).permutation(len(ds))[:num_samples]
        sub = Subset(ds, idx.tolist())
    return DataLoader(sub, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, drop_last=False, collate_fn=tri_collate_fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bi_ckpt', default='./Pretrain/bi_random/bimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--tri_ckpt', default='./Pretrain/tri_random/trimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--val_lmdb', default='./valid_set.lmdb')
    ap.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    ap.add_argument('--num_samples', type=int, default=8192)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--outdir', default='./probe_results')
    ap.add_argument('--outname', default='bivstri_metrics.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_shuffle', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False,
                              do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab, unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250)

    # =====================================================================
    # PART 1: pooled [CLS] global feature  →  eff_rank / CKA / dist-coverage
    # =====================================================================
    # ---- extract bimodal (text, prop) --------------------------------------
    print('=== BIMODAL (pooled feat) ===')
    bm = load_model(args.bi_ckpt, tokenizer, device)
    bf = bi_extract_all_feats(bm, tokenizer,
                           make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                                       args.num_workers, args.no_shuffle, args.seed),
                           device, args.num_samples)
    bi_text, bi_prop = bf['text_feat'], bf['prop_feat']
    del bm; gc.collect(); torch.cuda.empty_cache()

    # ---- extract trimodal (text, prop, dist) on the SAME molecules ----------
    print('=== TRIMODAL (pooled feat) ===')
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
    print(f'\nN={N}')

    # ---- (2) effective rank -------------------------------------------------
    print('\n=== effective rank (256-d feat) ===')
    er = {}
    for k in names:
        er[k] = effective_rank_metrics(reps[k])
        print(f'  {k:9s} eff_rank={er[k]["eff_rank"]:7.2f}  dim99={er[k]["dim99"]:4d}')

    # ---- (A) 스펙트럼(누적 에너지) 저장: bivstri_plot.py 가 JSON만으로 그림을 그리도록 --
    spectra = {}
    for k in names:
        s = _singular_values(reps[k]); lam = s ** 2
        spectra[k] = (np.cumsum(lam) / lam.sum()).tolist()

    # ---- (1) CKA matrix (incl. cross-model reshaping) -----------------------
    print('\n=== CKA(linear) matrix ===')
    M = np.zeros((5, 5))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = linear_cka(reps[a], reps[b])
    print('        ' + ' '.join(f'{n:>9s}' for n in names))
    for i, a in enumerate(names):
        print(f'{a:9s} ' + ' '.join(f'{M[i,j]:9.3f}' for j in range(5)))
    resh3_text = M[names.index('bi_text'), names.index('tri_text')]
    resh4_prop = M[names.index('bi_prop'), names.index('tri_prop')]
    print(f'\n  reshaping: CKA(bi_text,tri_text)={resh3_text:.3f}  '
          f'CKA(bi_prop,tri_prop)={resh4_prop:.3f}  (->1 = dist did not distort it)')

    # ---- (3) how much of dist is NEW over the bimodal pair ------------------
    print('\n=== dist coverage by existing modalities (R^2, shared / unique) ===')
    cov = {}
    for label, src in [('dist<-text', tri_text), ('dist<-prop', tri_prop),
                       ('dist<-text+prop', np.concatenate([tri_text, tri_prop], 1))]:
        r2 = cross_pred_r2(src, tri_dist, seed=args.seed)
        cov[label] = r2
        print(f'  {label:18s} R^2={r2:6.3f}   unique={1-r2:6.3f}')
    print(f'\n  => only {100*(1-cov["dist<-text+prop"]):.0f}% of the dist representation '
          f'is genuinely NEW beyond what bimodal (text+prop) already encodes.')

    # ---- (3b) same probe with prop as the TARGET ----------------------------
    # dist-coverage alone cannot tell whether a low unique-fraction is a property
    # of dist or just the generic level of text->X linear predictability in this
    # model. prop<-text is the control: it is measured twice, once inside the
    # trimodal model (where dist was also trained) and once inside the bimodal
    # model (where it was not), so the two numbers are directly comparable and
    # any gap is attributable to the third modality.
    print('\n=== prop coverage by text (R^2, shared / unique) — tri vs bi ===')
    prop_cov = {}
    for label, src, tgt in [('prop<-text (tri)', tri_text, tri_prop),
                            ('prop<-text (bi)', bi_text, bi_prop)]:
        r2 = cross_pred_r2(src, tgt, seed=args.seed)
        prop_cov[label] = r2
        print(f'  {label:18s} R^2={r2:6.3f}   unique={1-r2:6.3f}')
    d_prop = prop_cov['prop<-text (tri)'] - prop_cov['prop<-text (bi)']
    print(f'\n  => delta(tri - bi) = {d_prop:+.3f}   '
          f'({"tri" if d_prop > 0 else "bi"} couples text and prop more tightly)')

    # ---- build results (PART 1) --------------------------------------------
    results = {'N': N, 'eff_rank': er,
               'cka_matrix': {names[i]: {names[j]: float(M[i, j]) for j in range(5)} for i in range(5)},
               'reshaping': {'bi_text~tri_text': float(resh3_text),
                             'bi_prop~tri_prop': float(resh4_prop)},
               'dist_coverage_r2': cov,
               'prop_coverage_r2': prop_cov,
               'prop_coverage_r2_delta_tri_minus_bi': float(d_prop),
               'spectra_cumenergy': spectra}

    # =====================================================================
    # PART 2: layer-wise [CLS] cross-model CKA
    # =====================================================================
    # ---- extract bimodal ----------------------------------------------------
    print('\n=== BIMODAL (layer-wise) ===')
    bm = load_model(args.bi_ckpt, tokenizer, device)
    bf_l = extract_layerwise_cls(
        bm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del bm; gc.collect(); torch.cuda.empty_cache()

    # ---- extract trimodal (SAME molecules, identical permutation seed) -----
    print('=== TRIMODAL (layer-wise) ===')
    tm = load_model(args.tri_ckpt, tokenizer, device)
    tf_l = extract_layerwise_cls(
        tm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del tm; gc.collect(); torch.cuda.empty_cache()

    # ---- sanity: same architecture should give same #layers ----------------
    L_text = len(bf_l['text_layers'])
    L_prop = len(bf_l['prop_layers'])
    assert L_text == len(tf_l['text_layers']), \
        f"text layer count mismatch: bi={L_text} vs tri={len(tf_l['text_layers'])}"
    assert L_prop == len(tf_l['prop_layers']), \
        f"prop layer count mismatch: bi={L_prop} vs tri={len(tf_l['prop_layers'])}"
    N_lw = bf_l['text_layers'][0].shape[0]
    print(f'\nN={N_lw}, text hidden states={L_text} (= emb + {L_text-1} layers), '
          f'prop hidden states={L_prop} (= emb + {L_prop-1} layers)')

    # ---- per-layer cross-model CKA -----------------------------------------
    print('\nComputing layer-wise CKA ...')
    cka_text = [linear_cka(bf_l['text_layers'][k], tf_l['text_layers'][k])
                for k in range(L_text)]
    cka_prop = [linear_cka(bf_l['prop_layers'][k], tf_l['prop_layers'][k])
                for k in range(L_prop)]
    cka_text_proj = linear_cka(bf_l['text_feat'], tf_l['text_feat'])
    cka_prop_proj = linear_cka(bf_l['prop_feat'], tf_l['prop_feat'])

    # ---- print table -------------------------------------------------------
    print('\n=== layer-wise CKA(bi, tri) — same modality, same molecule, two ckpts ===')
    print(f'{"idx":>5s}  {"text":>7s}  {"prop":>7s}     interpretation of idx')
    L_max = max(L_text, L_prop)
    for k in range(L_max):
        label = 'emb' if k == 0 else f'L{k}'
        t = f'{cka_text[k]:7.3f}' if k < L_text else '   ---'
        p = f'{cka_prop[k]:7.3f}' if k < L_prop else '   ---'
        if k == 0:
            note = '   embedding output (after token embed + posn)'
        else:
            note = f'   transformer layer {k} output [CLS]'
        print(f'{label:>5s}  {t}  {p}{note}')
    print(f'{"proj":>5s}  {cka_text_proj:7.3f}  {cka_prop_proj:7.3f}'
          f'   post-projection L2-normalized InfoNCE feat')

    # ---- merge PART 2 into the SAME results dict (덮어쓰지 않고 추가) -------
    results.update({
        'L_text_hidden_states': L_text,
        'L_prop_hidden_states': L_prop,
        'cka_text_per_layer': [float(x) for x in cka_text],
        'cka_prop_per_layer': [float(x) for x in cka_prop],
        'cka_text_post_proj': float(cka_text_proj),
        'cka_prop_post_proj': float(cka_prop_proj),
    })

#    # =====================================================================
#    # PART 3: SHARED FUSION STACK — text_encoder layers [fusion_layer:]
#    # =====================================================================
#    # The unimodal probe above stops at the branch point. But pv2smiles /
#    # smiles2pv both continue THROUGH these cross-attention layers, so this is
#    # the last unmeasured segment between the encoder and the task metric.
#    # Only text<->prop pairings are probed: they are the only cross-attention
#    # directions the bimodal model also has, so the comparison stays fair.
#    print('\n=== BIMODAL (fusion stack) ===')
#    bm = load_model(args.bi_ckpt, tokenizer, device)
#    bf_f = extract_fusion_cls(
#        bm, tokenizer,
#        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
#                    args.num_workers, args.no_shuffle, args.seed),
#        device, args.num_samples)
#    del bm; gc.collect(); torch.cuda.empty_cache()
#
#    print('=== TRIMODAL (fusion stack) ===')
#    tm = load_model(args.tri_ckpt, tokenizer, device)
#    tf_f = extract_fusion_cls(
#        tm, tokenizer,
#        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
#                    args.num_workers, args.no_shuffle, args.seed),
#        device, args.num_samples)
#    del tm; gc.collect(); torch.cuda.empty_cache()
#
#    FUS = bf_f['fusion_layer_index']
#    assert tf_f['fusion_layer_index'] == FUS, 'fusion_layer mismatch between ckpts'
#    L_fus = len(bf_f['t_given_p'])
#    assert len(tf_f['t_given_p']) == L_fus, 'fusion layer count mismatch'
#
#    print(f'\nfusion_layer={FUS}, fusion hidden states={L_fus} '
#          f'(= branch point + {L_fus-1} fusion layers)')
#
#    cka_fus_tp = [linear_cka(bf_f['t_given_p'][k], tf_f['t_given_p'][k])
#                  for k in range(L_fus)]
#    cka_fus_pt = [linear_cka(bf_f['p_given_t'][k], tf_f['p_given_t'][k])
#                  for k in range(L_fus)]
#
#    # Consistency check: fusion index 0 is the *input* to the first fusion layer,
#    # i.e. the unimodal tower's last_hidden_state. It must reproduce the last
#    # point of PART 2. A mismatch means the two probes disagree on tokenization
#    # or masking and the curves must NOT be concatenated in the figure.
#    d_t = abs(cka_fus_tp[0] - cka_text[-1])
#    d_p = abs(cka_fus_pt[0] - cka_prop[-1])
#    print(f'  branch-point check  text: fusion[0]={cka_fus_tp[0]:.4f} vs '
#          f'unimodal[-1]={cka_text[-1]:.4f}  (|d|={d_t:.4f})')
#    print(f'  branch-point check  prop: fusion[0]={cka_fus_pt[0]:.4f} vs '
#          f'unimodal[-1]={cka_prop[-1]:.4f}  (|d|={d_p:.4f})')
#    if max(d_t, d_p) > 1e-3:
#        print('  [WARN] branch point disagrees between the unimodal and fusion '
#              'probes — do not read the two curves as one continuous path.')
#
#    print('\n=== fusion-stack CKA(bi, tri) — shared text_encoder layers ===')
#    print(f'{"idx":>6s}  {"t<-p":>7s}  {"p<-t":>7s}     interpretation')
#    for k in range(L_fus):
#        note = ('   branch point (= unimodal output, also feeds proj)' if k == 0
#                else f'   fusion layer {FUS+k-1} output [CLS]')
#        print(f'{f"F{FUS+k}":>6s}  {cka_fus_tp[k]:7.3f}  {cka_fus_pt[k]:7.3f}{note}')
#
#    print('\n  reading guide:')
#    print('  - rising toward 1.0 with depth = fusion re-converges the two models;')
#    print('    the encoder-level divergence is absorbed and task parity is expected')
#    print('  - flat / falling = bi and tri reach equal task scores through '
#          'genuinely different fusion computations')
#
#    results.update({
#        'fusion_layer_index': int(FUS),
#        'L_fusion_hidden_states': int(L_fus),
#        'cka_fusion_text_given_prop_per_layer': [float(x) for x in cka_fus_tp],
#        'cka_fusion_prop_given_text_per_layer': [float(x) for x in cka_fus_pt],
#    })

    # ---- save (한 번만) ----------------------------------------------------
    with open(f'{args.outdir}/{args.outname}', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nsaved metrics -> {args.outdir}/{args.outname}')

if __name__ == '__main__':
    main()
