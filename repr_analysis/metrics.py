"""Similarity / sharing measures between two representations."""
import numpy as np
from sklearn.linear_model import Ridge


def linear_cka(X, Y):
    """Linear CKA in [0,1]. X,Y: (N,d), centered internally."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    return float((xty ** 2).sum() /
                 (np.sqrt((xtx ** 2).sum()) * np.sqrt((yty ** 2).sum()) + 1e-12))


def cross_pred_r2(X, Y, alpha=1.0, frac=0.5, seed=0):
    """Fraction of Y's variance linearly reconstructable from X (ridge, held-out)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    ntr = int(N * frac)
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-12)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    muy, sdy = Y[tr].mean(0), np.maximum(Y[tr].std(0), 1e-12)
    Ytr, Yte = (Y[tr] - muy) / sdy, (Y[te] - muy) / sdy
    model = Ridge(alpha=alpha).fit(Xtr, Ytr)
    Yp = model.predict(Xte)
    ss_res = ((Yte - Yp) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))
