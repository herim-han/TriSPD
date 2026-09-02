import os
import torch

from .config import build_config

_MOMENTUM_PREFIXES = ('property_encoder_m', 'text_encoder_m', 'dist_encoder_m',
                      'property_proj_m', 'text_proj_m', 'dist_proj_m')


def _load(model, ckpt_path, device, verbose=True):
    ck = torch.load(ckpt_path, map_location='cpu')
    state = ck['state_dict'] if 'state_dict' in ck else ck
    for key in list(state.keys()):
        if '_unk' in key:
            state[key.replace('_unk', '_mask')] = state.pop(key)
    msg = model.load_state_dict(state, strict=False)
    if verbose:
        miss = [k for k in msg.missing_keys
                if not k.startswith(_MOMENTUM_PREFIXES) and 'queue' not in k]
        print(f'  loaded {os.path.basename(ckpt_path)}: '
              f'missing(non-mom/queue)={len(miss)} unexpected={len(msg.unexpected_keys)}')
    return model.eval().to(device)


def load_model(ckpt_path, tokenizer, device, verbose=False):
    """Pick the architecture from the checkpoint filename: 'bimodal' in the
    basename -> SPMM_models.SPMM, otherwise the trimodal SPMM. Quiet by default:
    run_bi_vs_tri loads four checkpoints and reports only a summary."""
    if 'bimodal' in os.path.basename(ckpt_path):
        from SPMM_models import SPMM
    else:
        from trimodal_bert_models_v3 import SPMM
    return _load(SPMM(config=build_config(), tokenizer=tokenizer, no_train=True),
                 ckpt_path, device, verbose)
