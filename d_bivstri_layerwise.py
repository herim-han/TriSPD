import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import ValLMDBDataset, tri_collate_fn
from trimodal_bert_models_v3 import SPMM
from d_effrank_v1 import build_config
from d_cka_v1 import linear_cka


def load_model(ckpt_path, tokenizer, device):
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
        sub = ds
    else:
        idx = np.random.default_rng(seed).permutation(len(ds))[:num_samples]
        sub = Subset(ds, idx.tolist())
    return DataLoader(sub, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, drop_last=False, collate_fn=tri_collate_fn)


@torch.no_grad()
def extract_layerwise_cls(model, tokenizer, loader, device, num_samples):
    """
    Extract per-layer [CLS] hidden states from text and prop encoders, plus
    the final L2-normalized contrastive features (post-projection).

    Returns:
        {
          'text_layers':  list of (N, H) np.ndarray, one per layer
                          (length = num_text_layers + 1, includes embedding output)
          'prop_layers':  list of (N, H) np.ndarray, one per layer
          'text_feat':    (N, embed_dim) np.ndarray, L2-normalized, post-projection
          'prop_feat':    (N, embed_dim) np.ndarray, L2-normalized, post-projection
        }
    Mirrors d_effrank_v1.extract_all_feats for tokenization / masking choices.
    """
    text_layers, prop_layers = None, None       # initialized on first batch
    text_feats, prop_feats = [], []
    n = 0
    for prop, smiles, atom_pair, dist in loader:
        prop = prop.to(device)

        # --- text tokenization (same convention as d_effrank_v1) -----------
        text = list(smiles)
        tin = tokenizer(text, padding='longest', truncation=True, max_length=100,
                        return_tensors='pt').to(device)
        ids = tin.input_ids[:, 1:]
        mask = tin.attention_mask[:, 1:]

        # --- text encoder: full hidden-state stack -------------------------
        # text_encoder.bert is a HuggingFace BertModel with the SPMM 'mode' patch.
        # Setting output_hidden_states=True returns the embedding output plus
        # each transformer layer's output. We pull the [CLS] token from each.
        text_out = model.text_encoder.bert(
            ids, attention_mask=mask,
            return_dict=True, output_hidden_states=True,
            mode='text',
        )
        text_hidden = text_out.hidden_states  # tuple of (B, T, H), len = L_text+1
        text_cls_per_layer = [h[:, 0, :].float().cpu().numpy() for h in text_hidden]
        # post-projection InfoNCE feat (matches *_feat from d_effrank_v1)
        t_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]),
                             dim=-1)

        # --- prop encoder: full hidden-state stack (NO masking) ------------
        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        prop_out = model.property_encoder(
            inputs_embeds=properties,
            return_dict=True, output_hidden_states=True,
        )
        prop_hidden = prop_out.hidden_states
        prop_cls_per_layer = [h[:, 0, :].float().cpu().numpy() for h in prop_hidden]
        p_feat = F.normalize(model.property_proj(prop_out.last_hidden_state[:, 0, :]),
                             dim=-1)

        # accumulate (initialize per-layer buffers on first batch)
        if text_layers is None:
            text_layers = [[] for _ in range(len(text_cls_per_layer))]
            prop_layers = [[] for _ in range(len(prop_cls_per_layer))]
        for i, t in enumerate(text_cls_per_layer):
            text_layers[i].append(t)
        for i, p in enumerate(prop_cls_per_layer):
            prop_layers[i].append(p)
        text_feats.append(t_feat.float().cpu().numpy())
        prop_feats.append(p_feat.float().cpu().numpy())

        n += prop.size(0)
        print(f'  collected {n}/{num_samples}', end='\r')
        if n >= num_samples:
            break
    print()

    return {
        'text_layers': [np.concatenate(L, axis=0)[:num_samples] for L in text_layers],
        'prop_layers': [np.concatenate(L, axis=0)[:num_samples] for L in prop_layers],
        'text_feat':   np.concatenate(text_feats, axis=0)[:num_samples],
        'prop_feat':   np.concatenate(prop_feats, axis=0)[:num_samples],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bi_ckpt',  default='./Pretrain/bi_random/bimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--tri_ckpt', default='./Pretrain/tri_random/trimodal_pubchem100m_96_step=216260.ckpt')
    ap.add_argument('--val_lmdb', default='./valid_set.lmdb')
    ap.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    ap.add_argument('--num_samples', type=int, default=8192)
    ap.add_argument('--batch_size',  type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--outdir', default='./probe_results')
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

    # ---- extract bimodal ----------------------------------------------------
    print('=== BIMODAL ===')
    bm = load_model(args.bi_ckpt, tokenizer, device)
    bf = extract_layerwise_cls(
        bm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del bm; gc.collect(); torch.cuda.empty_cache()

    # ---- extract trimodal (SAME molecules, identical permutation seed) -----
    print('=== TRIMODAL ===')
    tm = load_model(args.tri_ckpt, tokenizer, device)
    tf = extract_layerwise_cls(
        tm, tokenizer,
        make_loader(args.val_lmdb, args.num_samples, args.batch_size,
                    args.num_workers, args.no_shuffle, args.seed),
        device, args.num_samples)
    del tm; gc.collect(); torch.cuda.empty_cache()

    # ---- sanity: same architecture should give same #layers ----------------
    L_text = len(bf['text_layers'])
    L_prop = len(bf['prop_layers'])
    assert L_text == len(tf['text_layers']), \
        f"text layer count mismatch: bi={L_text} vs tri={len(tf['text_layers'])}"
    assert L_prop == len(tf['prop_layers']), \
        f"prop layer count mismatch: bi={L_prop} vs tri={len(tf['prop_layers'])}"
    N = bf['text_layers'][0].shape[0]
    print(f'\nN={N}, text hidden states={L_text} (= emb + {L_text-1} layers), '
          f'prop hidden states={L_prop} (= emb + {L_prop-1} layers)')

    # ---- per-layer cross-model CKA -----------------------------------------
    print('\nComputing layer-wise CKA ...')
    cka_text = [linear_cka(bf['text_layers'][k], tf['text_layers'][k])
                for k in range(L_text)]
    cka_prop = [linear_cka(bf['prop_layers'][k], tf['prop_layers'][k])
                for k in range(L_prop)]
    cka_text_proj = linear_cka(bf['text_feat'], tf['text_feat'])
    cka_prop_proj = linear_cka(bf['prop_feat'], tf['prop_feat'])

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

    print('\n  reading guide:')
    print('  - emb close to 1.0   = input embeddings barely changed by re-training')
    print('  - monotone decrease  = reshape builds up gradually with depth')
    print('  - cliff at last layer or at proj = reshape localized to the head '
          '(closer to "only the projection moved")')

    # ---- save --------------------------------------------------------------
    results = {
        'N': N,
        'L_text_hidden_states': L_text,
        'L_prop_hidden_states': L_prop,
        'cka_text_per_layer': [float(x) for x in cka_text],
        'cka_prop_per_layer': [float(x) for x in cka_prop],
        'cka_text_post_proj': float(cka_text_proj),
        'cka_prop_post_proj': float(cka_prop_proj),
    }
    with open(f'./layerwise_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
