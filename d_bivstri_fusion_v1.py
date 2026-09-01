"""
d_bivstri_fusion_v1.py
======================
Layer-wise CKA over the SHARED FUSION STACK (text_encoder layers
[fusion_layer : num_hidden_layers], i.e. 6..11 for config_bert.json).

Why this exists
---------------
d_bivstri_layerwise.py only probes the UNIMODAL towers (mode='text' -> layers
0..5, and the 6-layer property_encoder). But the zero-shot tasks do NOT stop
there -- they run the unimodal output through the fusion stack via cross
attention:

    pv2smiles_v4.py:29   text_encoder(text, encoder_hidden_states=prop_embeds)
    d_smiles2pv.py:24    text_encoder.bert(encoder_embeds=prop_embeds,
                                           encoder_hidden_states=text_embeds,
                                           mode='fusion')

So the fusion stack is the last unmeasured segment on the path from encoder to
task metric. If bi and tri agree on the tasks but their unimodal towers have
CKA ~0.6, the fusion stack is where the discrepancy either resolves or deepens.

Why the projection head is implicated even though tasks never call it
--------------------------------------------------------------------
`feat = normalize(proj(embeds[:,0,:]))` has no no_grad (extract_feature,
trimodal_bert_models_v3.py:41-43), so the InfoNCE gradient flows THROUGH proj
into the encoder. And ITM hard-negative mining samples from sim_dict, which is
built from `feat` (pair_matching, trimodal_bert_models_v3.py:460-461). So proj
shapes (a) the encoder by gradient and (b) the fusion stack by choosing which
negatives it trains against. The fusion path itself, however, consumes the
768-d `embeds`, never the 256-d `feat`.

Directions probed
-----------------
Only cross-attention pairings that exist in BOTH models, so the comparison is
apples-to-apples (the trimodal model additionally fuses dist, the bimodal one
cannot):

    t_given_p : Q=text, K/V=prop  -> pv2smiles / generation path
    p_given_t : Q=prop, K/V=text  -> smiles2pv / PV-prediction path

Alignment note
--------------
For a mode='fusion' call, xbert appends hidden_states BEFORE each layer in
[start_layer, output_layer) and once more at the end (xbert.py:575-624). Hence
hidden_states[0] is the INPUT to layer `fusion_layer`, which is bit-identical to
the unimodal tower's last_hidden_state. We keep index 0 on purpose:
  - it is the branch point shared with the InfoNCE head, and
  - CKA at index 0 must equal the unimodal probe's last layer -> free assert.
Plot x-coordinates are therefore `fusion_layer + k`, which makes the unimodal
and fusion curves join seamlessly at x = fusion_layer.
"""
import os
import sys

# Run from the repo root or from probe_test/ -- both layouts must import.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, 'probe_test')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch


@torch.no_grad()
def extract_fusion_cls(model, tokenizer, loader, device, num_samples):
    """
    Returns:
        {
          't_given_p': list of (N, H) np.ndarray -- [CLS] at each fusion layer,
                       Q=text conditioned on prop. len = n_fusion_layers + 1.
          'p_given_t': list of (N, H) np.ndarray -- Q=prop conditioned on text.
          'fusion_layer_index': int, the x-offset of index 0 of the lists above
                                (== config.fusion_layer).
        }
    Tokenization / CLS conventions mirror d_bivstri_layerwise.extract_layerwise_cls
    so the unimodal and fusion curves are directly concatenable.
    """
    import numpy as np

    tp_layers, pt_layers = None, None
    n = 0
    fusion_layer = int(model.text_encoder.bert.config.fusion_layer)

    for prop, smiles, atom_pair, dist in loader:
        prop = prop.to(device)

        # --- unimodal text tower (mode='text' -> layers 0..fusion_layer-1) ----
        text = list(smiles)
        tin = tokenizer(text, padding='longest', truncation=True, max_length=100,
                        return_tensors='pt').to(device)
        ids = tin.input_ids[:, 1:]
        text_atts = tin.attention_mask[:, 1:]
        text_embeds = model.text_encoder.bert(
            ids, attention_mask=text_atts, return_dict=True, mode='text',
        ).last_hidden_state

        # --- unimodal property tower (NO masking, matches the layerwise probe)
        pf = model.property_embed(prop.unsqueeze(2))
        properties = torch.cat(
            [model.property_cls.expand(prop.size(0), -1, -1), pf], dim=1)
        prop_embeds = model.property_encoder(
            inputs_embeds=properties, return_dict=True).last_hidden_state
        prop_atts = torch.ones(prop_embeds.size()[:-1], dtype=torch.long,
                               device=prop_embeds.device)

        # --- fusion, Q=text  K/V=prop  (pv2smiles direction) -----------------
        out_tp = model.text_encoder.bert(
            encoder_embeds=text_embeds, attention_mask=text_atts,
            encoder_hidden_states=prop_embeds, encoder_attention_mask=prop_atts,
            return_dict=True, output_hidden_states=True, mode='fusion',
        )
        # --- fusion, Q=prop  K/V=text  (smiles2pv direction) ------------------
        out_pt = model.text_encoder.bert(
            encoder_embeds=prop_embeds, attention_mask=prop_atts,
            encoder_hidden_states=text_embeds, encoder_attention_mask=text_atts,
            return_dict=True, output_hidden_states=True, mode='fusion',
        )

        tp_cls = [h[:, 0, :].float().cpu().numpy() for h in out_tp.hidden_states]
        pt_cls = [h[:, 0, :].float().cpu().numpy() for h in out_pt.hidden_states]

        if tp_layers is None:
            tp_layers = [[] for _ in range(len(tp_cls))]
            pt_layers = [[] for _ in range(len(pt_cls))]
        for i, h in enumerate(tp_cls):
            tp_layers[i].append(h)
        for i, h in enumerate(pt_cls):
            pt_layers[i].append(h)

        n += prop.size(0)
        print(f'  collected {n}/{num_samples}', end='\r')
        if n >= num_samples:
            break
    print()

    return {
        't_given_p': [np.concatenate(L, axis=0)[:num_samples] for L in tp_layers],
        'p_given_t': [np.concatenate(L, axis=0)[:num_samples] for L in pt_layers],
        'fusion_layer_index': fusion_layer,
    }
