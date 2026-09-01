import argparse
from tqdm import tqdm
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scheduler import create_scheduler
import random
import torch.nn as nn
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import os
import torch.nn.functional as F
from new_extract_features import DEFAULT_CACHE_DIR, load_cached_features, load_splits

ENCODER_KIND = 'text'            # SMILES only, layers 0-5 (shared with v3)

label_dict={
        'human_CL':0,
        'human_VDss':1,
        'human_fup':2
        }

def count(pred, true, min, max):
    lst = [abs(a/b) for a, b in zip(pred, true)]
    newlist = [x for x in lst if min <= x <= max]
    return (len(newlist)/len(lst)) *100

def calc_gmfe(pred, true):
    lst = [abs(np.log10(a/b)) for a, b in zip(pred, true)]
    mean_abs= np.mean(lst)
    return (10**mean_abs)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AnimalEmbedWithResidualGate(nn.Module):#integrated missing embedding vector
    def __init__(self, num_features=9, hidden_dim=256):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.animalpk_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_features)
        ])

        self.missing_embedding = nn.Parameter(torch.zeros(num_features, hidden_dim))

        self.missing_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        mask = ~torch.isnan(x)
        outputs = []
        for i in range(self.num_features):
            obs_mask = mask[:, i:i+1]
            xi = x[:, i:i+1]
            xi_safe = torch.where(obs_mask, xi, torch.zeros_like(xi))

            zi = self.animalpk_proj[i](xi_safe)
            miss_zi=self.missing_embedding[i].unsqueeze(0).expand(x.size(0), -1)

            zi = obs_mask.float() * zi + (1-obs_mask.float()) * miss_zi
            outputs.append(zi.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        pooled = output.mean(dim=1)
        gate = self.missing_gate(pooled)
        return pooled * (1.0 + gate)

class FrozenFeatureHead(nn.Module):
    """Everything that is still trained: the tail of SPMM_regressor.forward."""
    def __init__(self, text_width=768, hidden_width=512, num_animal=9,
                 feat_norm=False, head_dropout=0.0):
        super().__init__()
        self.feat_norm = nn.LayerNorm(text_width) if feat_norm else nn.Identity()
        self.fusion_proj = nn.Linear(text_width, hidden_width)
        self.animal_embed = AnimalEmbedWithResidualGate(num_animal, hidden_width)
        self.animal_norm = nn.LayerNorm(hidden_width)

        layers = [nn.Linear(hidden_width * 2, hidden_width * 4), nn.GELU()]
        if head_dropout > 0:
            layers.append(nn.Dropout(head_dropout))
        layers.append(nn.Linear(hidden_width * 4, 1))
        self.reg_head = nn.Sequential(*layers)

    def forward(self, feat, animal, value, eval=False):
        fusion_output = self.fusion_proj(self.feat_norm(feat))
        animal_embed = self.animal_embed(animal)
        animal_embed = self.animal_norm(animal_embed)
        vl_embeddings = torch.cat([fusion_output, animal_embed], axis=1)
        pred = self.reg_head(vl_embeddings).squeeze(-1)
        if eval:    return pred
        loss = nn.MSELoss()(pred, value)
        assert torch.isfinite(pred).all(), 'pred has Nan/Inf'
        assert torch.isfinite(value).all(), 'target has Nan/Inf'
        return loss

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, warmup_mode='step0'):
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 20
    num_steps = max(1, len(data_loader))

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    running_loss, n_batches = 0.0, 0
    for i, (feat, animal, target) in enumerate(tqdm_data_loader):
        if epoch == 0 and warmup_mode == 'gradual':
            scheduler.step(warmup_steps * i / max(1, num_steps - 1))

        optimizer.zero_grad()
        feat = feat.to(device, non_blocking=True)
        animal = animal.to(device, non_blocking=True)
        value = target.to(device, non_blocking=True)
        loss = model(feat, animal, value)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.3e}')

        if epoch == 0 and warmup_mode == 'step0':   # baseline v2 behaviour
            scheduler.step(0)

    return running_loss / max(1, n_batches)

@torch.no_grad()
def predict(model, feats, animals, values, device, batch_size=256):
    """Batched eval straight off the cached tensors -- no DataLoader, so the
    global RNG stream (and hence run-to-run reproducibility) is untouched."""
    model.eval()
    preds = []
    for s in range(0, feats.size(0), batch_size):
        f = feats[s:s + batch_size].to(device, non_blocking=True)
        a = animals[s:s + batch_size].to(device, non_blocking=True)
        v = values[s:s + batch_size].to(device, non_blocking=True)
        preds.append(model(f, a, v, eval=True).detach().cpu().numpy())
    return np.concatenate(preds, axis=0), values.numpy()

def compute_metrics(log_preds, log_answers, target_name):
    log_RMSE = root_mean_squared_error(log_answers, log_preds)
    log_R2   = r2_score(log_answers, log_preds)
    if target_name != 'human_fup':          # CL/VDss were trained in log10 space
        preds, answers = 10 ** log_preds, 10 ** log_answers
    else:                                    # fup stays linear (no log transform)
        preds, answers = log_preds, log_answers
    RMSE = root_mean_squared_error(answers, preds)
    R2   = r2_score(answers, preds)
    fold_error = count(preds, answers, 0.5, 2)
    gmfe = calc_gmfe(preds, answers)
    valid_df = pd.DataFrame({'preds': preds, 'true': answers})
    return log_RMSE, log_R2, RMSE, R2, fold_error, gmfe, valid_df

def ckpt_prefix(path):
    if 'trimodal' in path:
        return 'tri'
    if 'bimodal' in path:
        return 'bi'
    return 'biorg'

def make_masks(animal, ratios, n_masks, mask_seed):
    """{ratio: [bool (N, 9) drop-mask, ...]} for the VALIDATION animal PK only.

    Each of the N x 9 validation animal cells is dropped independently with
    probability `ratio`, so ratio 0.0 is the untouched data and 1.0 is "no animal
    PK at all". `n_masks` is how many times that dice roll is repeated per ratio:
    at 0.5 the score depends on WHICH cells vanish, so one draw would be noise.
    Ratios 0.0 and 1.0 have no randomness, so their draws are all identical.

    Drawn from a dedicated Generator so the global RNG -- and therefore training
    reproducibility for a given seed -- is untouched. Cells that were already NaN
    stay NaN. TRAIN is never masked.
    """
    rng = np.random.default_rng(mask_seed)
    return {r: [rng.random(tuple(animal.shape)) < r for _ in range(n_masks)] for r in ratios}


def train_one_head(feats, config, args, target_name, device, max_epoch, warmup_steps, seed,
                   masks=None):
    tr_f, tr_a, tr_v = feats['train']
    va_f, va_a, va_v = feats['valid']

    model = FrozenFeatureHead(text_width=tr_f.size(1), hidden_width=args.hidden_width,
                              num_animal=tr_a.size(1), feat_norm=args.feat_norm,
                              head_dropout=args.head_dropout).to(device)
    print('#trainable head parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # dedicated generator: shuffling never touches the global RNG
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(tr_f, tr_a, tr_v), batch_size=args.head_batch_size,
                              shuffle=True, drop_last=False, num_workers=0, generator=g)

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
    lr_scheduler, _ = create_scheduler(AttrDict(config['schedular']), optimizer)

    epoch_logpreds, epoch_train_logpreds = [], []
    answers_ref = train_answers_ref = None
    train_loss, lr_trace = [], []
    for epoch in range(0, max_epoch):
        lr_start = optimizer.param_groups[0]['lr']
        loss = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler,
                     warmup_mode=args.warmup_mode)
        lr_trace.append((lr_start, optimizer.param_groups[0]['lr']))
        train_loss.append(float(loss))

        lp, ans = predict(model, va_f, va_a, va_v, device)
        epoch_logpreds.append(lp)
        answers_ref = ans

        tr_lp, tr_ans = predict(model, tr_f, tr_a, tr_v, device)
        epoch_train_logpreds.append(tr_lp)
        train_answers_ref = tr_ans

        lr_scheduler.step(epoch + warmup_steps + 1)

    # === masked validation, at the LAST epoch only ===========================
    # That is the epoch every metric is reported at, and masking changes nothing
    # about training, so the model is scored again here rather than retrained.
    masked = {}
    if masks:
        for r, draws in masks.items():
            for mid, drop in enumerate(draws):
                a = va_a.clone()
                a[torch.from_numpy(drop)] = float('nan')
                masked[(r, mid)] = predict(model, va_f, a, va_v, device)[0]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (np.stack(epoch_logpreds, axis=0), answers_ref,
            np.stack(epoch_train_logpreds, axis=0), train_answers_ref, train_loss, lr_trace,
            masked)

def main(args, config):
    device = torch.device(args.device)

    # === Dataset === #
    # dataframes only: the SMILES never reach a tokenizer here, they are just what
    # the cached features are validated against (row count + fingerprint)
    print(f"Creating dataset: LOAD pbpk_db/{args.input}.csv")
    target_name = args.target_name
    splits = load_splits(args.input, target_name, args.imputer)
    print(f"train n={len(splits['train'])}, valid n={len(splits['valid'])}")

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    if config['schedular']['min_lr'] >= config['optimizer']['lr']:
        print(f"[warn] min_lr={config['schedular']['min_lr']:g} >= lr={config['optimizer']['lr']:g}: "
              f"the cosine has no range to decay, so the lr stays flat at min_lr after warmup. "
              f"Lower --min_lr (e.g. lr/100) or raise --lr.")
    if config['optimizer']['lr'] < 1e-4:
        print(f"[hint] lr={config['optimizer']['lr']:g} is a full-finetune lr. With the encoder frozen "
              f"only a randomly-initialised head is trained, which usually needs 1e-4 ~ 1e-3.")

    # === frozen encoder: load every checkpoint's cached embeddings ONCE, up front ===
    # (done before any seeding so the load cannot perturb the training RNG)
    feats_per_ckpt = []
    for ckpt in args.checkpoints:
        print(f'=== loading cached {ENCODER_KIND} embeddings: {ckpt} ===')
        feats_per_ckpt.append(load_cached_features(ENCODER_KIND, ckpt, args, splits, target_name))

    # === per-epoch curve logging === #
    os.makedirs(args.out_dir, exist_ok=True)
    # 'textonly' marks the SMILES-only embedding, so these CSVs never collide with
    # the fusion-embedding variant's when both write to the same --out_dir
    curve_tag = (f"frozen_textonly_{target_name}_lr{config['optimizer']['lr']:g}"
                 f"_bs{args.head_batch_size}_ep{max_epoch}")
    if args.tag:
        curve_tag += f'_{args.tag}'
    curve_path = os.path.join(args.out_dir, f'{curve_tag}.csv')
    mean_path  = os.path.join(args.out_dir, f'{curve_tag}_mean.csv')
    curve_rows = []

    # masks are built once from the valid animal tensor and reused for every seed
    # and every checkpoint, so all of them are scored on identical dropouts
    masks = make_masks(feats_per_ckpt[0]['valid'][1], args.ratios, args.n_masks, args.mask_seed)
    base_missing = torch.isnan(feats_per_ckpt[0]['valid'][1]).float().mean().item()
    print(f'[mask] valid animal cells already missing: {base_missing*100:.1f}%  |  '
          f'ratios={args.ratios}  n_masks={args.n_masks}  mask_seed={args.mask_seed}')
    mask_rows = []

    results = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        print(f'seed num: {seed}')
        start_time = time.time()

        per_model_masked = []
        per_model_logpreds, answers_ref, per_model_loss = [], None, {}
        per_model_train_logpreds, train_answers_ref = [], None
        model_key_order, lr_trace = [], None
        for i, (ckpt, feats) in enumerate(zip(args.checkpoints, feats_per_ckpt)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            cudnn.benchmark = True

            model_key = f'{i}_{ckpt_prefix(ckpt)}'
            model_key_order.append(model_key)

            print(f"Training @@@@@ {target_name} @@@@@ head on frozen embeddings from {ckpt}")
            lp, ans, tr_lp, tr_ans, tl, lr_trace, masked = train_one_head(
                feats, config, args, target_name, device, max_epoch, warmup_steps, seed,
                masks=masks,
            )
            per_model_masked.append(masked)

            if answers_ref is not None:
                assert np.allclose(ans, answers_ref), "validation order/answers mismatch across models"
            answers_ref = ans
            if train_answers_ref is not None:
                assert np.allclose(tr_ans, train_answers_ref), "train order/answers mismatch across models"
            train_answers_ref = tr_ans
            per_model_logpreds.append(lp)                 # (E, N) test
            per_model_train_logpreds.append(tr_lp)        # (E, M) train
            per_model_loss[model_key] = tl

        ens_logpreds = np.mean(np.stack(per_model_logpreds, axis=0), axis=0)
        ens_train_logpreds = np.mean(np.stack(per_model_train_logpreds, axis=0), axis=0)
        loss_curve = np.mean(np.stack([per_model_loss[k] for k in model_key_order], axis=0), axis=0)

        for epoch in range(ens_logpreds.shape[0]):
            tr_mse = mean_squared_error(train_answers_ref, ens_train_logpreds[epoch])
            te_mse = mean_squared_error(answers_ref, ens_logpreds[epoch])
            tr_rmse, tr_r2, _, _, tr_fold, tr_gmfe, _ = compute_metrics(ens_train_logpreds[epoch], train_answers_ref, target_name)
            te_rmse, te_r2, _, _, te_fold, te_gmfe, _ = compute_metrics(ens_logpreds[epoch], answers_ref, target_name)
            print(
                f'[epoch {epoch}] opt_loss={loss_curve[epoch]:.4f}'
                f'""mse/R2/2-FE"" TRAIN/{tr_mse:.3f}/{tr_r2:.2f}/{tr_fold:.1f}/'
                f'TEST/{te_mse:.3f}/{te_r2:.2f}/{te_fold:.1f}'
            )
            curve_rows.append({
                'seed': seed, 'epoch': epoch,
                'lr_start': lr_trace[epoch][0], 'lr_end': lr_trace[epoch][1],
                'train_loss': tr_mse, 'test_loss': te_mse,
                'train_2FE': tr_fold, 'test_2FE': te_fold,
                'opt_loss': loss_curve[epoch],
                'train_logRMSE': tr_rmse, 'test_logRMSE': te_rmse,
                'train_logR2': tr_r2, 'test_logR2': te_r2,
                'train_GMFE': tr_gmfe, 'test_GMFE': te_gmfe,
            })

        curve_df = pd.DataFrame(curve_rows)
#        curve_df.to_csv(curve_path, index=False)
        curve_df.drop(columns='seed').groupby('epoch').mean().reset_index().to_csv(mean_path, index=False)

        best_epoch = ens_logpreds.shape[0] - 1        # last epoch, fixed a priori
        log_rmse, log_r2, rmse, r2, fold_error, gmfe, vdf = \
            compute_metrics(ens_logpreds[best_epoch], answers_ref, target_name)
        best_results = [log_rmse, log_r2, fold_error, gmfe]
        print(f'Reporting LAST epoch ({best_epoch}): {best_results}')
        results.append(best_results)

        # masked validation for this seed: members ensembled, then mask draws averaged
        for r in args.ratios:
            mp = np.mean([np.mean([m[(r, mid)] for m in per_model_masked], axis=0)
                          for mid in range(args.n_masks)], axis=0)
            mm = compute_metrics(mp, answers_ref, target_name)
            mask_rows.append(dict(seed=seed, ratio=r, logRMSE=mm[0], logR2=mm[1],
                                  fold_error_2x=mm[4], GMFE=mm[5]))
            print(f'  [mask ratio {r:.2f}] LAST epoch: '
                  f'[{mm[0]:.4f}, {mm[1]:.4f}, {mm[4]:.4f}, {mm[5]:.4f}]')

        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print('Training time {}'.format(total_time_str))

    print(f'per-epoch curves -> {curve_path}')
    print(f'seed-averaged    -> {mean_path}')

    if not results:
        print('No best results recorded across seeds; nothing to aggregate.')
        return
    results = np.array(results)
    print(results)
    mean = results.mean(axis=0)
    std  = results.std(axis=0)
    # order: [log_RMSE, log_R2, 2-fold error, gmfe]
    print(f"{mean[0]:.2f}±{std[0]:.2f}")
    print(f"{mean[1]:.2f}±{std[1]:.2f}")
    print(f"{mean[2]:.2f}±{std[2]:.2f}")
    print(f"{mean[3]:.2f}±{std[3]:.2f}")

    # === masked-validation summary =========================================
    mask_df = pd.DataFrame(mask_rows)
    mask_path = os.path.join(args.out_dir, f'{curve_tag}_maskratio.csv')
#    mask_df.to_csv(mask_path, index=False)
    print(f'\n=== valid animal PK masking ({len(results)} seeds x {args.n_masks} mask draws) ===')
    print('order: log_RMSE, log_R2, 2-fold error, GMFE   (mean±std across seeds)')
    for r, g in mask_df.groupby('ratio'):
        v = g[['logRMSE', 'logR2', 'fold_error_2x', 'GMFE']]
        m, s = v.mean(), v.std(ddof=0)
        print(f'  ratio {r:.2f}: {m.logRMSE:.2f}±{s.logRMSE:.2f}  {m.logR2:.2f}±{s.logR2:.2f}  '
              f'{m.fold_error_2x:.2f}±{s.fold_error_2x:.2f}  {m.GMFE:.2f}±{s.GMFE:.2f}')
#    print(f'seed별 마스킹 결과 -> {mask_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+',
                        default=['./Pretrain/trimodal_pubchem100m_96_step=173008.ckpt',
                                 './Pretrain/trimodal_pubchem100m_96_step=216260.ckpt'],
                        help='One or more pretrained checkpoints to ensemble. Only used to look '
                             'up each member\'s cache file -- the .ckpt itself is never read.')
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt',
                        help='part of the cache key (no tokenizer is built here)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--imputer', action='store_true')
    parser.add_argument('--name', default='bace', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='kept for CLI parity; head training uses --head_batch_size')
    parser.add_argument('--target_name', default='human_CL', type=str)
    parser.add_argument('--input', default='src-iwata-pk', type=str)
    parser.add_argument('--num_seeds', default=100, type=int)
    parser.add_argument('--out_dir', default='./results_0728_frozen', type=str)
    # ---------------- frozen-encoder probe options ---------------- #
    parser.add_argument('--head_batch_size', default=128, type=int,
                        help='batch size for head training on the cached embeddings')
    parser.add_argument('--cache_dir', default=DEFAULT_CACHE_DIR, type=str,
                        help='where new_extract_features.py wrote the .npz caches')
    parser.add_argument('--extract_batch_size', default=64, type=int,
                        help='unused here; kept for CLI parity (see new_extract_features.py)')
    parser.add_argument('--max_len', default=100, type=int,
                        help='SMILES token truncation length -- part of the cache key')
    parser.add_argument('--hidden_width', default=512, type=int)
    parser.add_argument('--feat_norm', action='store_true',
                        help='LayerNorm the cached embedding before fusion_proj')
    parser.add_argument('--head_dropout', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.02, type=float)
    parser.add_argument('--warmup_mode', default='step0', choices=['step0', 'gradual'],
                        help="step0 = baseline v2 (lr pinned at warmup_lr through epoch 0); "
                             "gradual = ramp warmup_lr -> lr across epoch 0")
    parser.add_argument('--tag', default='', type=str)
    # ---------------- validation animal-PK masking ---------------- #
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.0],
                        help='probability of dropping each VALID animal cell. 0.0 = untouched '
                             'data, 1.0 = no animal PK. TRAIN is never masked. Default [0.0] '
                             'keeps the original behaviour; pass e.g. 0 0.25 0.5 0.75 1.0 to sweep.')
    parser.add_argument('--n_masks', default=20, type=int,
                        help='independent mask draws per ratio, averaged. Ratios 0.0 and 1.0 '
                             'have no randomness, so extra draws there change nothing.')
    parser.add_argument('--mask_seed', default=1234, type=int,
                        help='masks come from this alone, so every seed/checkpoint sees the same '
                             'dropouts and the training RNG is unaffected')

    args = parser.parse_args()
    cls_config = {
        'batch_size_train': args.batch_size,
        'batch_size_test': 16,
        'bert_config_text': './config_bert.json',
        'bert_config_property': './config_bert_property.json',
        'bert_config_dist': './config_bert_dist.json',
        'schedular': {'sched': 'cosine', 'lr': args.lr, 'epochs': args.epoch, 'min_lr': args.min_lr,
                      'decay_rate': 1, 'warmup_lr': 0.5e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': args.lr, 'weight_decay': args.weight_decay}
    }
    main(args, cls_config)
