"""Linear probes with honest (grouped) cross-validation.

All probes here are ridge regressions solved through one SVD per fold, so sweeping the ridge
penalty is nearly free. Group-aware splitting matters: here queries from the same sequence share
its posterior, so folds split by *sequence* rather than by *query* prevent a probe from scoring
well by memorising sequences.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _folds(n: int, groups: Optional[np.ndarray], n_splits: int, seed: int):
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.permutation(n)
        for k in range(n_splits):
            te = idx[k::n_splits]
            tr = np.setdiff1d(idx, te)
            yield tr, te
        return
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_splits = min(n_splits, len(uniq))
    for k in range(n_splits):
        test_groups = set(uniq[k::n_splits].tolist())
        te = np.array([i for i in range(n) if groups[i] in test_groups])
        tr = np.array([i for i in range(n) if groups[i] not in test_groups])
        yield tr, te


def r2_per_output(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1 - ss_res / ss_tot
    r2[ss_tot == 0] = np.nan
    return r2


def ridge_cv(
    X: np.ndarray,
    Y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    alphas: Sequence[float] = (1e-2, 1e-1, 1, 10, 100, 1000),
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Out-of-fold ridge predictions. Returns dict(pred, r2, alpha, r2_by_alpha).

    X: (n, d) features; Y: (n,) or (n, k) targets. Features are standardised inside each fold.
    The reported alpha is the one with the best mean out-of-fold R^2 (mildly optimistic;
    use ``nested_alpha=False`` results only as a ceiling and compare against baselines run the
    same way).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    squeeze = Y.ndim == 1
    if squeeze:
        Y = Y[:, None]
    n = X.shape[0]
    preds = {a: np.zeros_like(Y) for a in alphas}
    for tr, te in _folds(n, groups, n_splits, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ (Y[tr] - ym)
        for a in alphas:
            W = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
            preds[a][te] = Xte @ W + ym
    r2_by_alpha = {a: float(np.nanmean(r2_per_output(Y, p))) for a, p in preds.items()}
    best = max(r2_by_alpha, key=r2_by_alpha.get)
    pred = preds[best]
    r2 = r2_per_output(Y, pred)
    if squeeze:
        pred, r2 = pred[:, 0], r2[:1]
    return {"pred": pred, "r2": r2, "alpha": best, "r2_by_alpha": r2_by_alpha}


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> dict:
    """Fit ridge on all data. Returns weights in *raw* feature space: Y ~ X @ W + b."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Ws = Vt.T @ ((s / (s ** 2 + alpha))[:, None] * (U.T @ (Y - ym)))
    W = Ws / sd[:, None]
    b = ym - mu @ W
    return {"W": W, "b": b, "mu": mu, "sd": sd}


def permutation_baseline(X, Y, groups=None, n_perm: int = 3, seed: int = 0, **kw) -> float:
    """Mean R^2 after shuffling labels across groups (or rows). A probe must beat this."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y)
    vals = []
    for p in range(n_perm):
        if groups is None:
            Yp = Y[rng.permutation(len(Y))]
        else:
            uniq = np.unique(groups)
            perm = dict(zip(uniq, rng.permutation(uniq)))
            first = {g: np.where(groups == g)[0][0] for g in uniq}
            Yp = np.stack([Y[first[perm[g]]] for g in groups])
        vals.append(float(np.nanmean(ridge_cv(X, Yp, groups, seed=seed + p, **kw)["r2"])))
    return float(np.mean(vals))


def mlp_probe(X, Y, groups=None, hidden: int = 256, epochs: int = 300, lr: float = 3e-3, test_frac: float = 0.2,
              seed: int = 0, device: str = "cpu") -> dict:
    """One-hidden-layer MLP probe with a group-aware train/test split. Returns test R^2 and predictions.

    Use it next to the linear probe: a large MLP-over-linear gap means the information is present
    but not linearly arranged (as with Othello-GPT before the "mine/theirs" basis was found).
    """
    import torch

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    rng = np.random.default_rng(seed)
    if groups is None:
        te = rng.random(len(X)) < test_frac
    else:
        uniq = np.unique(groups)
        test_groups = set(rng.choice(uniq, size=max(1, int(len(uniq) * test_frac)), replace=False).tolist())
        te = np.array([g in test_groups for g in groups])
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=device)
    Yt = torch.tensor(Y, device=device)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    tr_idx = torch.tensor(np.nonzero(~te)[0], device=device)
    bs = 4096
    for ep in range(epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(Xt).cpu().numpy()
    return {"r2": r2_per_output(Y[te], pred[te]), "pred_test": pred[te], "y_test": Y[te], "test_mask": te}
