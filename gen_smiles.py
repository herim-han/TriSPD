import argparse
import torch
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from calc_property import calculate_property
from rdkit import Chem
import random
import numpy as np
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings(action='ignore')
import pandas as pd
import os


# ----------------------------------------------------------------------------- #
#  Decoding: stochastic beam search (ported from pv2smiles_single.py)
# ----------------------------------------------------------------------------- #
def beam_step(model, prop_embeds, text, beam_width, stochastic, temperature, top_k):
    """One decoding step for beam search.
    text: (B, T) current hypotheses. prop_embeds: (B, Lp, feat) matching condition.
    Returns (logp, idx), each (B, beam_width): the log-prob and token id of the
    beam_width continuations proposed for every hypothesis.
      stochastic=True  -> multinomial WITHOUT replacement  (== stochastic beam search)
      stochastic=False -> deterministic top-k              (== plain beam search)"""
    text_atts = torch.where(text == 0, 0, 1)                                  # 0 is [PAD]
    prop_att = torch.ones(prop_embeds.size()[:-1], dtype=torch.long, device=prop_embeds.device)
    logits = model.text_encoder(text,
                                attention_mask=text_atts,
                                encoder_hidden_states=prop_embeds,
                                encoder_attention_mask=prop_att,
                                return_dict=True,
                                is_decoder=True,
                                return_logits=True,
                                )[:, -1, :]                                    # (B, vocab)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    p = torch.softmax(logits, dim=-1)
    if stochastic:
        idx = torch.multinomial(p, num_samples=beam_width, replacement=False)  # (B, W)
    else:
        idx = torch.topk(p, beam_width, dim=-1).indices                        # (B, W)
    logp = torch.log(torch.gather(p, 1, idx) + 1e-12)                          # (B, W)
    return logp, idx


@torch.no_grad()
def beam_generate(model, prop_embeds_1, beam_width, max_len, stochastic, temperature, top_k):
    """Beam search for ONE condition. prop_embeds_1: (1, Lp, feat).
    Keeps `beam_width` running hypotheses, collects up to beam_width**2 finished ones,
    and returns a SINGLE decoded SMILES (top-scoring beam; a random finished beam when
    stochastic). This is the batched-per-condition version of the loop in
    pv2smiles_single.py, with prop_embeds expanded to the beam width."""
    device = prop_embeds_1.device
    tok = model.tokenizer
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    prop_k = prop_embeds_1.expand(beam_width, -1, -1)                          # (W, Lp, feat) view

    # --- first step from [CLS]: 1 hypothesis -> beam_width hypotheses ---
    first = torch.full((1, 1), cls_id, dtype=torch.long, device=device)
    logp, idx = beam_step(model, prop_embeds_1, first, beam_width, stochastic, temperature, top_k)  # (1, W)
    beams = torch.cat([torch.full((beam_width, 1), cls_id, dtype=torch.long, device=device),
                       idx.view(beam_width, 1)], dim=1)                        # (W, 2)
    beam_logp = logp.view(beam_width)                                          # (W,)

    finished = []                                                             # list of (score, seq)
    for _ in range(max_len):
        logp, idx = beam_step(model, prop_k, beams, beam_width, stochastic, temperature, top_k)  # (W, W)
        cand_score = beam_logp[:, None] + logp                                # (W, W)
        cand_seq = torch.cat([beams.unsqueeze(1).repeat(1, beam_width, 1),
                              idx.unsqueeze(-1)], dim=-1)                      # (W, W, T+1)

        # harvest any hypothesis that just emitted [SEP]
        if (idx == sep_id).any():
            for e in (idx == sep_id).nonzero(as_tuple=False):
                finished.append((cand_score[e[0], e[1]].item(), cand_seq[e[0], e[1]]))
                cand_score[e[0], e[1]] = -1e9                                  # remove from the pool
            if len(finished) >= beam_width ** 2:
                break

        # keep the best `beam_width` continuations
        beam_logp, flat = torch.topk(cand_score.flatten(), beam_width)        # (W,)
        rows,cols = flat // beam_width, flat % beam_width
        beams = torch.stack([cand_seq[r, c] for r, c in zip(rows.tolist(), cols.tolist())], dim=0)

    if not finished:                                                          # never hit [SEP] within max_len
        finished = [(beam_logp[j].item(), beams[j]) for j in range(beam_width)]
    finished.sort(key=lambda x: x[0], reverse=True)
    finished = finished[:beam_width]
    seq = random.choice(finished)[1] if stochastic else finished[0][1]        # 1-sample per condition

    cdd = tok.convert_tokens_to_string(
        tok.convert_ids_to_tokens(seq[:-1])).replace('[CLS]', '')             # drop trailing [SEP]
    return cdd


# ----------------------------------------------------------------------------- #
#  Property encoding (per-row masking)
# ----------------------------------------------------------------------------- #
def encode_properties(model, prop_norm, prop_mask, device):
    """prop_norm: (B, 53) normalized values. prop_mask: (B, 53) PER-ROW, 0=given, 1=masked."""
    B = prop_norm.size(0)
    prop = prop_norm.to(device, non_blocking=True)
    property1 = model.property_embed(prop.unsqueeze(2))                       # B*53*feat
    property_unk = model.property_mask.expand(B, property1.size(1), -1)       # B*53*feat
    mask_expand = prop_mask.to(device).unsqueeze(2).expand(B, -1, property1.size(2))  # per-row
    prop_masked = property1 * (1 - mask_expand) + property_unk * mask_expand
    properties = torch.cat([model.property_cls.expand(B, -1, -1), prop_masked], dim=1)
    return model.property_encoder(inputs_embeds=properties, return_dict=True).last_hidden_state


@torch.no_grad()
def generate_with_property(model, prop_targets, prop_mask, beam_width=2, n_sample=1,
                           batch_size=64, max_len=100, stochastic=True, temperature=1.0, top_k=0):
    """Encode all conditions (per-row masks) in batches, then beam-decode n_sample molecules per row.
    prop_targets: (N,53) raw values. prop_mask: (N,53) per-row, 0=given, 1=masked.
    beam_width : search width of ONE beam search (quality/breadth of a single decode).
    n_sample  : how many molecules to emit per condition (independent stochastic beam-search runs).
    Returns a flat list of length N*n_sample, ordered row0*n_sample, row1*n_sample, ...
    (i.e. aligned with prop_targets.repeat_interleave(n_sample, dim=0))."""
    device = model.device
    model.eval()
    with open('./normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)
    prop_norm_all = (prop_targets - mean) / std
    N = prop_norm_all.size(0)

    # encode conditions in batches (cheap, batched transformer forward)
    embeds = []
    for start in range(0, N, batch_size):
        pe = encode_properties(model, prop_norm_all[start:start + batch_size],
                               prop_mask[start:start + batch_size], device)   # (b, Lp, feat)
        embeds.append(pe)
    prop_embeds_all = torch.cat(embeds, dim=0)                                # (N, Lp, feat)

    # beam search decodes one condition at a time; repeat n_sample times per condition
    candidates = []
    for i in tqdm(range(N), leave=False):
        pe = prop_embeds_all[i:i + 1]
        for _ in range(n_sample):                                            # n_sample independent runs
            candidates.append(beam_generate(model, pe, beam_width, max_len,
                                            stochastic, temperature, top_k))
    return candidates


# ----------------------------------------------------------------------------- #
#  Evaluation (per-row mask)
# ----------------------------------------------------------------------------- #
def canonical(smiles):
    m = Chem.MolFromSmiles(smiles)
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


@torch.no_grad()
def metric_eval(prop_targets, cand, mask, sources):
    """PER-ROW evaluation. mask (N,53): 0=given, 1=masked. RMSE on GIVEN props only.
    Returns (rmse, validity, uniqueness, novelty)."""
    with open('./normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)

    per_row_rmse, valid_canon, valid_src = [], [], []
    for i in range(len(cand)):
        try:
            prop_cdd = calculate_property(cand[i])                            # raises if invalid
        except Exception:
            continue
        given = (mask[i] == 0)
        if given.sum() == 0:
            continue
        n_ref = (prop_targets[i] - mean) / std
        n_cdd = (prop_cdd - mean) / std
        sq = (n_ref - n_cdd) ** 2
        per_row_rmse.append(torch.sqrt(sq[given].mean()).item())
        valid_canon.append(canonical(cand[i]))
        valid_src.append(sources[i])

    v = len(valid_canon)
    if v == 0:
        return float('nan'), 0.0, float('nan'), float('nan')
    rmse = float(np.mean(per_row_rmse))
    validity = v / len(cand)
    uniqueness = len(set(valid_canon)) / v
    novelty = float(np.mean([g != s for g, s in zip(valid_canon, valid_src)]))
    return rmse, validity, uniqueness, novelty


# ----------------------------------------------------------------------------- #
#  Masking / data
# ----------------------------------------------------------------------------- #
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_row_masks(N, k, n_prop=53):
    """Per-row random mask: each row exposes exactly k given properties, masks the rest.
    Returns (N, n_prop) with 0=given, 1=masked. Uniform over the n_prop properties (same
    spirit as the Bernoulli(0.5) pretraining masking, but with a fixed given-count k)."""
    given_idx = torch.rand(N, n_prop).argsort(dim=1)[:, :k]
    mask = torch.ones(N, n_prop)
    mask.scatter_(1, given_idx, 0.0)
#    print('mask: \n',mask.long())
#    exit(-1)
    return mask


def load_test_molecules(path, n_mols, seed=0):
    """Read SMILES (one per line), subsample n_mols, compute the 53-dim PV for each.
    Returns prop_targets (M,53) raw values and sources (list of M canonical SMILES)."""
    smiles_list = [ln.strip() for ln in open(path).readlines() if ln.strip()]
    rng = random.Random(seed)
    if n_mols and n_mols < len(smiles_list):
        smiles_list = rng.sample(smiles_list, n_mols)
    prop_rows, sources = [], []
    for smi in tqdm(smiles_list, desc='calc_property', leave=False):
        try:
            prop = calculate_property(smi)
            can = canonical(smi)
            if can is None:
                continue
        except Exception:
            continue
        prop_rows.append(prop)
        sources.append(can)
    prop_targets = torch.stack(prop_rows, dim=0)
    print(f"Loaded {prop_targets.size(0)} valid molecules from {path}")
    return prop_targets, sources


# ----------------------------------------------------------------------------- #
def main(args, config):
    device = torch.device(args.device)
    if 'bimodal' in os.path.basename(args.checkpoint):
        from SPMM_models import SPMM
    else:
        from trimodal_bert_models import SPMM

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    print("Creating model")
    model = SPMM(config=config, tokenizer=tokenizer, no_train=True)
    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    prop_targets, sources = load_test_molecules(args.test_file, args.n_mols, seed=args.mol_seed)
    N = prop_targets.size(0)

    k_list = [int(k) for k in str(args.k_list).split(',')]                    # single int or comma list
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    stochastic = not args.greedy
    mode = f"stochastic beam (W={args.beam_width}, T={args.temperature}, top_k={args.top_k})" if stochastic \
        else f"deterministic beam (W={args.beam_width})"

    print("=" * 60)
    print(f"Random-mask k-sweep | mols={N} | k={k_list} | seeds={seeds} | {mode}")
    print("=" * 60)

    records = []
    for k in k_list:
        for seed in seeds:
            set_seed(seed)                                                    # controls masks AND sampling
            mask = sample_row_masks(N, k)                                     # (N,53) per-row, 0=given 1=masked
            samples = generate_with_property(
                model, prop_targets, mask,
                beam_width=args.beam_width, n_sample=args.n_sample,
                batch_size=args.batch_size, max_len=args.max_len,
                stochastic=stochastic, temperature=args.temperature, top_k=args.top_k)
            # expand each row's target/mask/source to match its n_sample generations
            # repeat_interleave = x.expand().clone()
            pt_rep = prop_targets.repeat_interleave(args.n_sample, dim=0)
            mask_rep = mask.repeat_interleave(args.n_sample, dim=0)
            src_rep = [s for s in sources for _ in range(args.n_sample)]
            rmse, validity, uniqueness, novelty = metric_eval(pt_rep, samples, mask_rep, src_rep)
            records.append({'k': k, 'seed': seed, 'rmse': rmse,
                            'validity': validity, 'uniqueness': uniqueness, 'novelty': novelty})
            print(f"  k={k:3d} seed={seed}: rmse={rmse:.4f} valid={validity:.4f} "
                  f"uniq={uniqueness:.4f} novel={novelty:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)

    print("=" * 60)
    print(f"Summary: metric vs #given-properties k  (mean +- std over {len(seeds)} seeds)")
    print(f"{'k':>4} | {'rmse':>16} | {'validity':>16} | {'uniqueness':>16} | {'novelty':>16}")
    for k in k_list:
        sub = df[df['k'] == k]
        cells = []
        for m in ['rmse', 'validity', 'uniqueness', 'novelty']:
            vals = sub[m].to_numpy()
            cells.append(f"{np.nanmean(vals):.4f}+-{np.nanstd(vals):.4f}")
        print(f"{k:>4} | " + " | ".join(f"{c:>16}" for c in cells))
    print("=" * 60)
    print(f"Per-(k,seed) results saved to '{args.out_csv}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_file', default='./test_sample.txt')
    parser.add_argument('--n_mols', default=200, type=int)
    parser.add_argument('--mol_seed', default=0, type=int)
    parser.add_argument('--k_list', default='1,3,5,10,20,40')
    parser.add_argument('--n_seeds', default=5, type=int, help='mask/decoding seeds per k')
    parser.add_argument('--seed', default=50, type=int, help='first seed; seeds=[seed, seed+n_seeds)')
    parser.add_argument('--beam_width', default=2, type=int, help='beam search width of a single decode (k in the reference)')
    parser.add_argument('--n_sample', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for property encoding')
    parser.add_argument('--max_len', default=150, type=int)
    parser.add_argument('--greedy', action='store_true',
                        help='deterministic beam (default: stochastic beam search)')
    parser.add_argument('--temperature', default=1.5, type=float)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--out_csv', default='./pv2smiles_v4_results.csv')
    arg = parser.parse_args()

    cudnn.benchmark = True
    configs = {
        'embed_dim': 256,
        'property_width': 768,
        'dist_width': 768,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'bert_config_dist': './config_bert_dist.json',
    }
    main(arg, configs)
