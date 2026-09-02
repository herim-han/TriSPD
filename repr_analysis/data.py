import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, WordpieceTokenizer

from dataset import LMDBDataset, tri_collate_fn


def build_tokenizer(vocab_filename):
    tokenizer = BertTokenizer(vocab_file=vocab_filename, do_lower_case=False,
                              do_basic_tokenize=False, add_special_tokens=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=tokenizer.vocab, unk_token=tokenizer.unk_token,
        max_input_chars_per_word=250)
    return tokenizer


def make_loader(val_lmdb, num_samples, batch_size, num_workers, no_shuffle, seed):
    """Fixed-seed random subset. Two calls with the same seed select the same
    molecules, which is what makes the bi/tri comparison paired."""
    ds = LMDBDataset(val_lmdb)
    idx = np.random.default_rng(seed).permutation(len(ds))[:num_samples]
    sub = Subset(ds, idx.tolist())
    return DataLoader(sub, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, drop_last=False, collate_fn=tri_collate_fn)
