"""Effective-rank summaries of a representation's singular spectrum.

SPMM is trained with InfoNCE, so its embedding dimensions are not axis-aligned and
DeCUR's per-dimension diagnostic does not apply. These are the rotation-invariant
generalization: the singular spectrum of the centered embedding matrix.
"""
import numpy as np


def _singular_values(X, standardize=False, eps=1e-12):
    """Singular values of the centered (optionally z-scored) N x K matrix."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if standardize:
        sd = X.std(axis=0, keepdims=True)
        X = X / np.maximum(sd, eps)
    s = np.linalg.svd(X, compute_uv=False)
    return s


def effective_rank_metrics(X, standardize=False):
    """Rotation-invariant rank summaries.

      eff_rank   : Roy-Vetterli entropy rank  exp(-sum p_k log p_k),  p_k = s_k / sum s
      part_ratio : participation ratio        (sum s^2)^2 / sum s^4
      stable_rank: ||X||_F^2 / ||X||_2^2
      dim99/dim95: # components for 99%/95% of spectral energy
      K          : ambient dimension
    """
    s = _singular_values(X, standardize=standardize)
    s = s[s > 0]
    K = int(np.asarray(X).shape[1])
    if s.size == 0:
        return dict(eff_rank=0.0, part_ratio=0.0, stable_rank=0.0,
                    dim99=0, dim95=0, K=K, s_max=0.0)
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))
    lam = s ** 2
    part_ratio = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))
    stable_rank = float(lam.sum() / (lam.max() + 1e-12))
    energy = np.cumsum(lam) / lam.sum()
    dim99 = int(np.searchsorted(energy, 0.99) + 1)
    dim95 = int(np.searchsorted(energy, 0.95) + 1)
    return dict(eff_rank=eff_rank, part_ratio=part_ratio, stable_rank=stable_rank,
                dim99=dim99, dim95=dim95, K=K, s_max=float(s.max()))
