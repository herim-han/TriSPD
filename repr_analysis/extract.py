"""Feature extraction from a loaded SPMM, without the pretraining MPM/MDM masking
so the clean learned representation is read.

    extract_all_feats      pooled [CLS] + projected feats, text / prop / dist
    bi_extract_all_feats   same for a bimodal checkpoint (no dist branch)
    extract_layerwise_cls  per-layer [CLS] of the unimodal text and prop towers
"""
import numpy as np
import torch
import torch.nn.functional as F


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
        ids = tin.input_ids[:, 1:]
        mask = tin.attention_mask[:, 1:]

        t_embeds = model.text_encoder.bert(ids, attention_mask=mask,
                                           return_dict=True,
                                           mode='text').last_hidden_state
        t_cls = t_embeds[:, 0, :]
        t_feat = F.normalize(model.text_proj(t_cls), dim=-1)

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
        if n >= num_samples:
            break
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
        ids = tin.input_ids[:, 1:]
        mask = tin.attention_mask[:, 1:]

        t_embeds = model.text_encoder.bert(ids, attention_mask=mask,
                                           return_dict=True,
                                           mode='text').last_hidden_state
        t_cls = t_embeds[:, 0, :]
        t_feat = F.normalize(model.text_proj(t_cls), dim=-1)

        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        p_embeds = model.property_encoder(inputs_embeds=properties,
                                          return_dict=True).last_hidden_state
        p_cls = p_embeds[:, 0, :]
        p_feat = F.normalize(model.property_proj(p_cls), dim=-1)

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
        if n >= num_samples:
            break
    return {k: np.concatenate(v, axis=0)[:num_samples] for k, v in feats.items()}


@torch.no_grad()
def extract_layerwise_cls(model, tokenizer, loader, device, num_samples):
    """Per-layer [CLS] of the text and prop encoders plus the final L2-normalized
    contrastive feats. Returns 'text_layers' / 'prop_layers' (lists of (N,H),
    index 0 = embedding output) and 'text_feat' / 'prop_feat'. Tokenization and
    masking follow extract_all_feats."""
    text_layers, prop_layers = None, None
    text_feats, prop_feats = [], []
    n = 0
    for prop, smiles, atom_pair, dist in loader:
        prop = prop.to(device)

        text = list(smiles)
        tin = tokenizer(text, padding='longest', truncation=True, max_length=100,
                        return_tensors='pt').to(device)
        ids = tin.input_ids[:, 1:]
        mask = tin.attention_mask[:, 1:]

        text_out = model.text_encoder.bert(
            ids, attention_mask=mask,
            return_dict=True, output_hidden_states=True,
            mode='text',
        )
        text_cls_per_layer = [h[:, 0, :].float().cpu().numpy()
                              for h in text_out.hidden_states]
        t_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]),
                             dim=-1)

        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        prop_out = model.property_encoder(
            inputs_embeds=properties,
            return_dict=True, output_hidden_states=True,
        )
        prop_cls_per_layer = [h[:, 0, :].float().cpu().numpy()
                              for h in prop_out.hidden_states]
        p_feat = F.normalize(model.property_proj(prop_out.last_hidden_state[:, 0, :]),
                             dim=-1)

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
        if n >= num_samples:
            break

    return {
        'text_layers': [np.concatenate(L, axis=0)[:num_samples] for L in text_layers],
        'prop_layers': [np.concatenate(L, axis=0)[:num_samples] for L in prop_layers],
        'text_feat':   np.concatenate(text_feats, axis=0)[:num_samples],
        'prop_feat':   np.concatenate(prop_feats, axis=0)[:num_samples],
    }
