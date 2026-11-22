"""Fair baselines for in-context forecasting of exponential-kernel Hawkes processes.

The oracle knows the true (mu, alpha, beta), which no forecaster can. The fair comparisons use only
the events seen so far:

  Bayes   the best possible forecaster given the training prior: average the exact forecast over
          all parameter values, weighted by how well each explains the history (importance sampling
          with K draws from the prior). This is the Bayes-optimal in-context predictor.
  ML      the classical approach: fit the parameters by maximum likelihood on the history (best of
          the same K candidate draws, i.e. constrained to the prior's support), then forecast.
  Poisson rate = events so far / time so far.

Evaluated at fixed context lengths m: after m events, forecast the interval to event m + 1.
Both kernels: exponential (exact recursion, NumPy) and power-law (all pairs, PyTorch, GPU if available).
"""
from __future__ import annotations

import numpy as np
import torch

from .simulate import oracle_bin_probs, oracle_bin_probs_powerlaw, poisson_bin_probs, sample_params
from .task import kernel_of, make_batch


def _loglik_checkpoints(times: np.ndarray, mu, alpha, beta, checkpoints) -> tuple[dict, dict]:
    """For one event sequence and K parameter vectors: log-likelihood of the first m events and the
    excitation just after event m - 1, at each checkpoint m."""
    K = len(mu)
    ll = np.zeros(K)
    E_after = np.zeros(K)
    prev = 0.0
    out_ll, out_E = {}, {}
    cps = set(checkpoints)
    for k in range(max(checkpoints)):
        dt = times[k] - prev
        decay = np.exp(-beta * dt)
        E_before = E_after * decay
        ll += np.log(mu + E_before) - mu * dt - (E_after / beta) * (1.0 - decay)
        E_after = E_before + alpha * beta
        prev = times[k]
        if k + 1 in cps:
            out_ll[k + 1] = ll.copy()
            out_E[k + 1] = E_after.copy()
    return out_ll, out_E


def bin_probs_powerlaw_torch(hist: np.ndarray, p: dict, edges: np.ndarray, device=None, max_elems: float = 1e7) -> np.ndarray:
    """Next-interval bin probabilities after the last event of ``hist`` for each parameter draw in p, (K, n_bins)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.as_tensor(hist[-1] - hist, dtype=torch.float64, device=device)          # (m,)
    ef = torch.as_tensor(edges[1:-1], dtype=torch.float64, device=device)              # finite inner edges
    K, m = len(p["mu"]), len(hist)
    kc = max(1, int(max_elems // (m * len(ef))))
    out = np.zeros((K, len(edges) - 1))
    for a in range(0, K, kc):
        sl = slice(a, min(K, a + kc))
        mu, al, c, g = (torch.as_tensor(np.asarray(p[k])[sl], dtype=torch.float64, device=device) for k in ("mu", "alpha", "c", "gamma"))
        Phi = lambda x: al[:, None, None] * (1 - (1 + x / c[:, None, None]) ** (1 - g[:, None, None]))  # noqa: E731
        comp = mu[:, None] * ef[None] + (Phi(x0[None, :, None] + ef[None, None]) - Phi(x0[None, :, None])).sum(1)
        S = torch.cat([torch.ones_like(comp[:, :1]), torch.exp(-comp), torch.zeros_like(comp[:, :1])], 1)
        P = torch.clamp(S[:, :-1] - S[:, 1:], min=1e-12)
        out[sl] = (P / P.sum(1, keepdim=True)).cpu().numpy()
    return out


@torch.no_grad()
def compare(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, checkpoints=(2, 4, 8, 16, 32, 64, 128),
            n_prior: int = 20000, weight_floor: float = 1e-10) -> dict:
    """Cross-entropy (nats) at each context length for model, oracle, Bayes, ML plug-in and Poisson."""
    if kernel_of(dcfg) == "powerlaw":
        return compare_powerlaw(model, dcfg, edges, n_seqs, seed, device, checkpoints, n_prior, weight_floor)
    model.eval()
    N = int(dcfg["n_events"])
    checkpoints = [m for m in checkpoints if m <= N - 1]
    rng = np.random.default_rng(seed + 5)
    pr = dcfg.get("params", {})
    prior = sample_params(rng, n_prior, mu=tuple(pr.get("mu", (0.2, 2.0))), alpha=tuple(pr.get("alpha", (0.0, 0.9))),
                          beta=tuple(pr.get("beta", (0.5, 10.0))))
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 404], edges, with_truth=True)
    logits = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device))["logits"]
    logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
    ce = {k: {m: [] for m in checkpoints} for k in ("model", "oracle", "bayes", "ml", "poisson")}
    ess = {m: [] for m in checkpoints}
    for s in range(n_seqs):
        lls, Es = _loglik_checkpoints(b["times64"][s], prior["mu"], prior["alpha"], prior["beta"], checkpoints)
        for m in checkpoints:
            i = m - 1  # position after the m-th event; it predicts the interval to event m + 1
            y = int(b["labels"][s, i])
            ce["model"][m].append(-logp[s, i, y])
            P = oracle_bin_probs(b["mu"][s], b["E_after"][s, i], b["beta"][s], edges)
            ce["oracle"][m].append(-np.log(P[y]))
            ll = lls[m]
            w = np.exp(ll - ll.max())
            w /= w.sum()
            keep = w > weight_floor * w.max()
            Pk = oracle_bin_probs(prior["mu"][keep], Es[m][keep], prior["beta"][keep], edges)
            ce["bayes"][m].append(-np.log(max((w[keep] / w[keep].sum()) @ Pk[:, y], 1e-300)))
            ess[m].append(1.0 / np.sum(w ** 2))
            j = int(np.argmax(ll))
            Pml = oracle_bin_probs(prior["mu"][j], Es[m][j], prior["beta"][j], edges)
            ce["ml"][m].append(-np.log(Pml[y]))
            Pp = poisson_bin_probs(np.array(m / b["times64"][s, i]), edges)
            ce["poisson"][m].append(-np.log(Pp[y]))
    return _summarise(ce, ess, checkpoints, n_prior)


def _summarise(ce: dict, ess: dict, checkpoints, n_prior: int) -> dict:
    rows = []
    for m in checkpoints:
        arr = {k: np.array(v[m]) for k, v in ce.items()}
        d = arr["model"] - arr["bayes"]
        rows.append({"context": m, **{k: float(a.mean()) for k, a in arr.items()},
                     "model_minus_bayes": float(d.mean()), "model_minus_bayes_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                     "model_minus_ml": float((arr["model"] - arr["ml"]).mean()),
                     "model_minus_ml_se": float((arr["model"] - arr["ml"]).std(ddof=1) / np.sqrt(len(d))),
                     "median_ess": float(np.median(ess[m])), "n": int(len(d))})
    return {"rows": rows, "n_prior": n_prior}


@torch.no_grad()
def compare_powerlaw(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, checkpoints=(2, 4, 8, 16, 32, 64, 128),
                     n_prior: int = 20000, weight_floor: float = 1e-10) -> dict:
    """The same comparison for the power-law kernel (prior over mu, alpha, c, gamma)."""
    from .posterior import loglik_powerlaw, prior_draws

    model.eval()
    dev = device if isinstance(device, str) else str(device)
    N = int(dcfg["n_events"])
    checkpoints = [m for m in checkpoints if m <= N - 1]
    prior = prior_draws(dcfg, n_prior, seed + 5)
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 404], edges, with_truth=True)
    logits = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device))["logits"]
    logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
    ce = {k: {m: [] for m in checkpoints} for k in ("model", "oracle", "bayes", "ml", "poisson")}
    ess = {m: [] for m in checkpoints}
    for s in range(n_seqs):
        t = b["times64"][s]
        ll_all = loglik_powerlaw(t, prior, max(checkpoints), dev)
        orc = oracle_bin_probs_powerlaw(t[None, :max(checkpoints) + 1], b["mu"][s:s + 1], b["alpha"][s:s + 1], b["c"][s:s + 1],
                                        b["gamma"][s:s + 1], edges)[0]
        for m in checkpoints:
            i = m - 1
            y = int(b["labels"][s, i])
            ce["model"][m].append(-logp[s, i, y])
            ce["oracle"][m].append(-np.log(orc[i, y]))
            ll = ll_all[:, i]
            w = np.exp(ll - ll.max())
            w /= w.sum()
            keep = w > weight_floor * w.max()
            sub = {k: np.asarray(v)[keep] for k, v in prior.items()}
            Pk = bin_probs_powerlaw_torch(t[:m], sub, edges, dev)
            ce["bayes"][m].append(-np.log(max((w[keep] / w[keep].sum()) @ Pk[:, y], 1e-300)))
            ess[m].append(1.0 / np.sum(w ** 2))
            j = int(np.argmax(ll))
            Pml = bin_probs_powerlaw_torch(t[:m], {k: np.asarray(v)[j:j + 1] for k, v in prior.items()}, edges, dev)[0]
            ce["ml"][m].append(-np.log(Pml[y]))
            Pp = poisson_bin_probs(np.array(m / t[i]), edges)
            ce["poisson"][m].append(-np.log(Pp[y]))
    return _summarise(ce, ess, checkpoints, n_prior)
