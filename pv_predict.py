"""SMILES-to-PV generation and evaluation.

The architecture is chosen from the checkpoint filename, the same rule
repr_analysis uses: 'bimodal' in the basename -> SPMM_models.SPMM (no 3D branch),
otherwise trimodal_bert_models_v3.SPMM. The trimodal path additionally enriches
the text embeddings with the dist branch before generating the property vector;
the bimodal model has no dist weights, so it feeds the raw text embeddings.
"""
import argparse
import os
import random
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, WordpieceTokenizer
from sklearn.metrics import r2_score
from tqdm import tqdm

from dataset import LMDBDataset, tri_collate_fn


def is_bimodal_checkpoint(path):
    return 'bimodal' in os.path.basename(path)


def build_model(checkpoint, tokenizer, config):
    """Instantiate the architecture the checkpoint was trained with."""
    if is_bimodal_checkpoint(checkpoint):
        from SPMM_models import SPMM
    else:
        from trimodal_bert_models_v3 import SPMM
    return SPMM(config=config, tokenizer=tokenizer, no_train=True)


def generate(model, prop_input, kv_embeds, kv_atts):
    prop_embeds = model.property_encoder(inputs_embeds=prop_input, return_dict=True).last_hidden_state
    prob_atts = torch.ones(prop_input.size()[:-1], dtype=torch.long).to(prop_input.device)
    token_output = model.text_encoder.bert(encoder_embeds=prop_embeds,
                                           attention_mask=prob_atts,
                                           encoder_hidden_states=kv_embeds,
                                           encoder_attention_mask=kv_atts,
                                           return_dict=True,
                                           is_decoder=True,
                                           mode='fusion',
                                           ).last_hidden_state
    pred = model.property_mtr_head(token_output).squeeze(-1)[:, -1]
    return pred.unsqueeze(1)


def encode_dist(model, atom_pair, dist):
    dist_feature = model.dist_embed_layer(atom_pair, dist)
    cls = model.dist_cls.expand(dist_feature.size(0), -1, -1)
    distances = torch.cat([cls, dist_feature], dim=1)
    dist_embeds = model.dist_encoder(inputs_embeds=distances, return_dict=True).last_hidden_state
    return dist_embeds


def enrich_text_with_dist(model, text_embeds, text_atts, dist_embeds, dist_atts):
    return model.text_encoder.bert(encoder_embeds=text_embeds,
                                   attention_mask=text_atts,
                                   encoder_hidden_states=dist_embeds,
                                   encoder_attention_mask=dist_atts,
                                   return_dict=True,
                                   mode='fusion',
                                   ).last_hidden_state


@torch.no_grad()
def pv_generate(model, data_loader, use_dist):
    with open('./normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)
    device = model.device
    tokenizer = model.tokenizer
    model.eval()
    print(f"SMILES-to-PV generation ({'trimodal, dist-enriched' if use_dist else 'bimodal'})...")

    reference, candidate = [], []
    for (prop, text, atom_pair, dist) in tqdm(data_loader, total=len(data_loader)):
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_atts = text_input.attention_mask[:, 1:]
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:],
                                              attention_mask=text_atts,
                                              return_dict=True,
                                              mode='text').last_hidden_state

        if use_dist:
            atom_pair = atom_pair.to(device, non_blocking=True)
            dist = dist.to(device, non_blocking=True)
            dist_embeds = encode_dist(model, atom_pair, dist)
            dist_atts = torch.ones(dist_embeds.size()[:-1], dtype=torch.long, device=device)
            kv_embeds = enrich_text_with_dist(model, text_embeds, text_atts, dist_embeds, dist_atts)
        else:
            kv_embeds = text_embeds

        prop_input = model.property_cls.expand(len(text), -1, -1)
        prediction = []
        for _ in range(53):
            output = generate(model, prop_input, kv_embeds, text_atts)
            prediction.append(output)
            output = model.property_embed(output.unsqueeze(2))
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(prop.size(0)):
            reference.append(prop[i].cpu())
            candidate.append(prediction[i].cpu())
    return reference, candidate


@torch.no_grad()
def metric_eval(ref, cand):
    with open('./normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    mean, std = norm
    mse = []
    n_mse = []
    rs, cs = [], []
    for i in range(len(ref)):
        r = (ref[i] * std) + mean
        c = (cand[i] * std) + mean
        rs.append(r)
        cs.append(c)
        mse.append((r - c) ** 2)
        n_mse.append((ref[i] - cand[i]) ** 2)
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0)).squeeze()
    n_mse = torch.stack(n_mse, dim=0)
    n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    print('mean of 53 properties\' normalized RMSE:', n_rmse.mean().item())

    rs = torch.stack(rs)
    cs = torch.stack(cs).squeeze()
    r2 = []
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:, i], cs[:, i]))
    r2 = np.array(r2)
    print('mean r^2 coefficient of determination:', r2.mean().item())


def main(args, config):
    device = torch.device(args.device)

    seed = 42
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset")
    dataset_test = LMDBDataset(args.input_file)
    print(f'{args.input_file} len={len(dataset_test)}')
    test_loader = DataLoader(dataset_test,
                             batch_size=config['batch_size_test'],
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=tri_collate_fn)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    use_dist = not is_bimodal_checkpoint(args.checkpoint)
    print(f"Creating model ({'trimodal' if use_dist else 'bimodal'}, from {os.path.basename(args.checkpoint)})")
    model = build_model(args.checkpoint, tokenizer, config)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    print("=" * 50)
    r_test, c_test = pv_generate(model, test_loader, use_dist)
    metric_eval(r_test, c_test)
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--vocab_filename', default='new_vocab_spe_496.txt')
    parser.add_argument('--input_file', default='./valid_set.lmdb')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'property_width': 768,
        'dist_width': 768,
        'batch_size_test': 256,
        'bert_config_text': './src/config/config_bert.json',
        'bert_config_property': './src/config/config_bert_property.json',
        'bert_config_dist': './src/config/config_bert_dist.json',
    }
    main(args, config)
