"""Does the model hold the Bayesian posterior over the parameters, rather than the parameters themselves?

The best possible forecaster (Bayes) cannot know the true parameters; it knows the posterior given the events so
far. If the transformer is Bayes-optimal (round 4), the natural hypothesis is that its residual stream encodes
posterior quantities. For every probe position, importance sampling over K prior draws gives the posterior mean
and standard deviation of each parameter. Per parameter and layer:

  R2 true       probe target = the true parameter (what earlier rounds reported)
  R2 posterior  probe target = the posterior mean E[theta | events so far]
  R2 sd         probe target = the posterior standard deviation (does the model know how unsure it is?)
  ceiling       R2 of the posterior mean as a predictor of the true parameter = the most that ANY function of
                the history can explain. R2 true / ceiling = the share of the available information the layer holds.

Exponential kernel: exact recursive likelihood (NumPy). Power-law kernel: pairwise likelihood in PyTorch (GPU if
available), chunked over prior draws.
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.probes import ridge_cv
from .simulate import sample_params, sample_params_powerlaw
from .task import kernel_of


def prior_draws(dcfg: dict, K: int, seed) -> dict:
    """K draws from the training prior, with the probe parameterisation (log_mu, alpha, log_beta[, gamma])."""
    rng = np.random.default_rng(seed)
    pr = dcfg.get("params", {})
    if kernel_of(dcfg) == "powerlaw":
        p = sample_params_powerlaw(rng, K, mu=tuple(pr.get("mu", (0.2, 2.0))), alpha=tuple(pr.get("alpha", (0.0, 0.9))),
                                   c=tuple(pr.get("c", (0.1, 2.0))), gamma=tuple(pr.get("gamma", (1.5, 3.0))))
        p["beta"] = 1.0 / p["c"]
    else:
        p = sample_params(rng, K, mu=tuple(pr.get("mu", (0.2, 2.0))), alpha=tuple(pr.get("alpha", (0.0, 0.9))),
                          beta=tuple(pr.get("beta", (0.5, 10.0))))
    return p


def loglik_exp(times: np.ndarray, p: dict, upto: int) -> np.ndarray:
    """(K, upto): log-likelihood of the first m events (m = 1..upto), compensator up to the m-th event."""
    mu, alpha, beta = p["mu"], p["alpha"], p["beta"]
    K = len(mu)
    ll = np.zeros(K)
    E = np.zeros(K)
    prev = 0.0
    out = np.zeros((K, upto))
    for k in range(upto):
        dt = times[k] - prev
        decay = np.exp(-beta * dt)
        Eb = E * decay
        ll += np.log(mu + Eb) - mu * dt - (E / beta) * (1.0 - decay)
        E = Eb + alpha * beta
        prev = times[k]
        out[:, k] = ll
    return out


def loglik_powerlaw(times: np.ndarray, p: dict, upto: int, device=None, max_elems: float = 1e7) -> np.ndarray:
    """(K, upto) for phi(x) = alpha (gamma - 1) / c (1 + x / c)^-gamma, exact (all pairs), chunked over K."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(times[:upto], dtype=torch.float64, device=device)
    gap = t[:, None] - t[None, :]                          # gap[k, j] = t_k - t_j
    low = torch.tril(torch.ones(upto, upto, dtype=torch.bool, device=device), -1)
    gap = torch.where(low, gap, torch.zeros_like(gap))
    K = len(p["mu"])
    kc = max(1, int(max_elems // (upto * upto)))
    out = np.zeros((K, upto))
    for a in range(0, K, kc):
        sl = slice(a, min(K, a + kc))
        mu, al, c, g = (torch.as_tensor(p[k][sl], dtype=torch.float64, device=device)[:, None, None]
                        for k in ("mu", "alpha", "c", "gamma"))
        base = 1 + gap[None] / c
        phi = torch.where(low[None], al * (g - 1) / c * base ** (-g), torch.zeros((), dtype=torch.float64, device=device))
        Phi = torch.where(low[None], al * (1 - base ** (1 - g)), torch.zeros((), dtype=torch.float64, device=device))
        lam = mu[:, :, 0] + phi.sum(-1)                    # intensity at each event (just before it)
        comp = mu[:, :, 0] * t[None] + Phi.sum(-1)         # compensator up to each event
        out[sl] = (torch.cumsum(torch.log(lam), -1) - comp).cpu().numpy()
    return out


def loglik(dcfg: dict, times: np.ndarray, p: dict, upto: int, device=None) -> np.ndarray:
    if kernel_of(dcfg) == "powerlaw":
        return loglik_powerlaw(times, p, upto, device)
    return loglik_exp(times, p, upto)


def _targets(dcfg: dict, p: dict) -> dict:
    t = {"log_mu": np.log(p["mu"]), "alpha": p["alpha"], "log_beta": np.log(p["beta"])}
    if kernel_of(dcfg) == "powerlaw":
        t["gamma"] = p["gamma"]
    return t


def posterior_stats(dcfg: dict, times: np.ndarray, positions: np.ndarray, K: int = 20000, seed=0, device=None) -> dict:
    """Posterior mean, sd and effective sample size after event i for every i in ``positions``.
    times: (B, N). Returns {name: {"mean": (B, Q), "sd": (B, Q)}, "ess": (B, Q)}."""
    p = prior_draws(dcfg, K, seed)
    th = _targets(dcfg, p)
    B = times.shape[0]
    upto = int(positions.max()) + 1
    res = {k: {"mean": np.zeros((B, len(positions))), "sd": np.zeros((B, len(positions)))} for k in th}
    ess = np.zeros((B, len(positions)))
    for s in range(B):
        ll = loglik(dcfg, times[s], p, upto, device)[:, positions]   # (K, Q): after events 0..i
        w = np.exp(ll - ll.max(0, keepdims=True))
        w /= w.sum(0, keepdims=True)
        ess[s] = 1.0 / (w ** 2).sum(0)
        for k, v in th.items():
            m1 = v @ w
            m2 = (v ** 2) @ w
            res[k]["mean"][s] = m1
            res[k]["sd"][s] = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    res["ess"] = ess
    return res


def _r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def posterior_probes(col: dict, dcfg: dict, K: int = 20000, seed=0, device=None) -> dict:
    """Probe each layer for the true parameter, its posterior mean and its posterior sd (see module docstring)."""
    post = posterior_stats(dcfg, col["times"], col["query_idx"], K=K, seed=seed, device=device)
    names = [k for k in post if k != "ess"]
    truth = {k: col["targets"][k] for k in names}
    rows = []
    ceiling = {k: _r2(truth[k], post[k]["mean"].reshape(-1)) for k in names}
    for li, X in enumerate(col["resid"]):
        row = {"layer": li}
        for k in names:
            row[f"{k}__true"] = float(ridge_cv(X, truth[k], groups=col["groups"], alphas=(0.1, 1, 10, 100))["r2"][0])
            row[f"{k}__posterior_mean"] = float(ridge_cv(X, post[k]["mean"].reshape(-1), groups=col["groups"],
                                                         alphas=(0.1, 1, 10, 100))["r2"][0])
            row[f"{k}__posterior_sd"] = float(ridge_cv(X, post[k]["sd"].reshape(-1), groups=col["groups"],
                                                       alphas=(0.1, 1, 10, 100))["r2"][0])
        rows.append(row)
    return {"rows": rows, "ceiling": ceiling, "params": names, "K": K,
            "median_ess": float(np.median(post["ess"])), "min_ess_median_by_seq": float(np.median(post["ess"].min(1)))}
