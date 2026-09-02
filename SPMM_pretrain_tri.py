import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed
import argparse
from pathlib import Path
from transformers import BertTokenizer, WordpieceTokenizer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import CSVLogger
import time
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import mmap
import joblib
from joblib import Parallel, delayed
from dataset import tri_collate_fn, ShuffledMultiLMDBDataset, LMDBDataset
import glob
import faulthandler
from datetime import datetime
torch.set_float32_matmul_precision('medium')
faulthandler.enable()
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING']='1'

def main(args, config):
    from trimodal_bert_models_v3 import SPMM

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

    all_dataset = LMDBDataset(args.input_file)
    data_loader = DataLoader(all_dataset,
                             batch_size=config['batch_size'], 
                             num_workers=args.num_workers, 
                             shuffle=False,# use already shuffled train/valid set
                             pin_memory=False,
                             drop_last=True,
                             collate_fn = tri_collate_fn)

    model = SPMM(config=config, tokenizer=tokenizer, loader_len=len(data_loader) // torch.cuda.device_count())
    model.enable_gradient_checkpointing()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)
    time = datetime.now().strftime('%Y%m%d')
    csv_logger = CSVLogger('lightning_csv_logs', name=f'trispd-bs-{config["batch_size"]}-dim-{config["property_width"]}-{time}')

    save_n_steps = int(len(all_dataset)/args.devices/config['batch_size']//10 )

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, 
                                                       filename=f'trimodal_pubchem100m_{config["batch_size"]}_'+'{step}',
                                                       every_n_train_steps=save_n_steps,
                                                       save_top_k=-1,
                                                       )
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=args.devices, 
                         precision='16-mixed', 
                         max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], 
                         strategy=DDPStrategy(find_unused_parameters=True), 
                         limit_val_batches=0.,
                         reload_dataloaders_every_n_epochs=1,
                         )

    trainer.logger = csv_logger
    print('start model.fit')
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--pkl', default=None)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain')
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--vocab_filename', default='./new_vocab_spe_496.txt')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debugging', action='store_true', default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--bert', action='store_true', default=False)
    args = parser.parse_args()

    pretrain_config = {
                       'property_width': 768,
                       'dist_width': 768,
                       'embed_dim': 256,
                       'batch_size': args.batch_size,#192
                       'temp': 0.07,
                       'mlm_probability': 0.15,
                       'queue_size': 24960,
                       'momentum': 0.995,
                       'alpha': 0.4,
                       'bert_config_text': './src/config/config_bert.json',
                       'bert_config_property': './src/config/config_bert_property.json',
                       'bert_config_dist': './src/config/config_bert_dist.json',
                       'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 10, 'min_lr': 5e-5,
                                     'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
                       'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02},
                       }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)
