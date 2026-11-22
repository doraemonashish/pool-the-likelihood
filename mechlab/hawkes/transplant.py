"""Causal test of the parameter representation: "transplant" a Hawkes parameter.

The round-2 runs showed (i) alpha, log mu and log beta are linearly decodable, (ii) one early layer
needs long context while later layers only need recent events. The two-stage hypothesis: early
layers *estimate the parameters* from the whole history; later layers combine those estimates with
the last few events. If so, editing the parameter estimate should change the forecast exactly the
way the true process would change.

Method, per layer l and parameter p in {alpha, log_mu, log_beta}:
  1. Fit a linear map A (3 x d) from (alpha, log mu, log beta) to the residual stream after block l,
     on held-out sequences (late positions only).
  2. On new sequences, add  delta_p * A[p]  at every position after block l, where delta_p moves the
     parameter to a different value inside the training range.
  3. Compare the change in the model's implied log intensity with the change in the true log
     intensity if the same event history had been generated with the new parameter value.
Scores
  slope      regression of the model's change on the true change (1 = full transplant, 0 = no effect).
             Not parameter-specific: raising alpha, mu or beta all raise the intensity somewhere.
  pattern_p  within-sequence correlation (each sequence's mean change removed) between the model's
             change and the true-change pattern of parameter p: alpha matters right after bursts, mu in
             quiet spells, beta in how fast bursts fade. An edit is *specific* if its own parameter's
             pattern beats the other two. A random direction shows the model's generic sensitivity
             profile, which can resemble one parameter's pattern; it is the reference row.
"""
from __future__ import annotations

import numpy as np
import torch

from .task import collect, make_batch

PARAMS = ["alpha", "log_mu", "log_beta"]
RANGES = {"alpha": (0.02, 0.88), "log_mu": (np.log(0.2), np.log(2.0)), "log_beta": (np.log(0.5), np.log(10.0))}
STEPS = {"alpha": 0.3, "log_mu": 0.8, "log_beta": 1.0}


def excitation_after(times: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """E just after each event (B, N), for arbitrary parameters on a given event history."""
    B, N = times.shape
    E = np.zeros((B, N))
    e = np.zeros(B)
    prev = np.zeros(B)
    for i in range(N):
        e = e * np.exp(-beta * (times[:, i] - prev)) + alpha * beta
        E[:, i] = e
        prev = times[:, i]
    return E


def fit_maps(model, dcfg: dict, edges, n_seqs: int, seed, device, query_from: int) -> list[np.ndarray]:
    """Per layer (after block l, l = 0..L-1): A (3, d) with resid ~ c + [alpha, log mu, log beta] @ A."""
    col = collect(model, dcfg, edges, n_seqs, seed, device, query_from=query_from)
    V = np.stack([col["targets"]["alpha"], col["targets"]["log_mu"], col["targets"]["log_beta"]], 1)
    Vc = V - V.mean(0)
    maps = []
    for li in range(1, len(col["resid"])):
        X = col["resid"][li]
        A = np.linalg.solve(Vc.T @ Vc + 1e-3 * np.eye(3), Vc.T @ (X - X.mean(0)))
        maps.append(A)
    return maps


def _new_values(v: np.ndarray, p: str, rng) -> np.ndarray:
    lo, hi = RANGES[p]
    step = STEPS[p]
    sign = np.where(rng.random(len(v)) < 0.5, -1.0, 1.0)
    out = v + sign * step
    bad = (out < lo) | (out > hi)
    out[bad] = v[bad] - sign[bad] * step
    return np.clip(out, lo, hi)


def _within_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation after removing each sequence's (row's) mean."""
    x = (a - a.mean(1, keepdims=True)).ravel()
    y = (b - b.mean(1, keepdims=True)).ravel()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def intensity_gradients(times: np.ndarray, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, h: float = 1e-3) -> dict:
    """d log(lambda just after each event) / d(alpha, log mu, log beta), shape (B, N) each."""
    K = excitation_after(times, np.ones_like(alpha), beta)  # E = alpha * K
    lam = mu[:, None] + alpha[:, None] * K
    Kb = excitation_after(times, np.ones_like(alpha), beta * np.exp(h))
    lam_b = mu[:, None] + alpha[:, None] * Kb
    return {"alpha": K / lam, "log_mu": mu[:, None] / lam, "log_beta": (np.log(lam_b) - np.log(lam)) / h}


@torch.no_grad()
def transplant(model, dcfg: dict, edges, n_seqs: int, seed, device, query_from: int = 48, n_fit: int = 400,
               scales=(1.0,)) -> list[dict]:
    """Rows per (edited parameter, layer, method): slope vs the true change, and within-sequence pattern
    correlations with each parameter's true-change pattern (pattern_<p>). Specificity = the edited
    parameter's own pattern beats the other parameters' patterns, and a random edit shows no such preference."""
    model.eval()
    rng = np.random.default_rng(seed + 3)
    maps = fit_maps(model, dcfg, edges, n_fit, seed + 11, device, query_from)
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 2], edges, with_truth=True)
    feats = torch.from_numpy(b["feats"]).to(device)
    pos = torch.from_numpy(b["times"]).to(device)
    N = b["feats"].shape[1]
    q = np.arange(query_from, N - 1)
    width = edges[1] - edges[0]

    def implied(edit=None):
        lg = model(feats=feats, pos=pos, edit=edit)["logits"].float()
        p0 = torch.softmax(lg, -1)[..., 0].cpu().numpy()
        return np.log(-np.log1p(-np.clip(p0, 0, 1 - 1e-7)) / width + 1e-12)[:, q]

    base = implied()
    cur = {"alpha": b["alpha"], "log_mu": np.log(b["mu"]), "log_beta": np.log(b["beta"])}
    grads = {k: v[:, q] for k, v in intensity_gradients(b["times64"], b["mu"], b["alpha"], b["beta"]).items()}
    lam_old = np.exp(cur["log_mu"])[:, None] + excitation_after(b["times64"], cur["alpha"], np.exp(cur["log_beta"]))
    rows = []
    for p_i, p in enumerate(PARAMS):
        new = dict(cur)
        new[p] = _new_values(cur[p], p, rng)
        lam_new = np.exp(new["log_mu"])[:, None] + excitation_after(b["times64"], new["alpha"], np.exp(new["log_beta"]))
        d_oracle = (np.log(lam_new) - np.log(lam_old))[:, q]
        delta = (new[p] - cur[p])[:, None]  # (B, 1)
        sgn = np.sign(delta)
        patterns = {p2: sgn * g for p2, g in grads.items()}  # true-change pattern if p2 moved the same way
        for li, A in enumerate(maps):
            norm = np.linalg.norm(A[p_i])
            other = A[(p_i + 1) % 3]
            r = rng.standard_normal(A.shape[1])
            dirs = {"param_map": A[p_i], "ctrl_random": r / np.linalg.norm(r) * norm,
                    "ctrl_other_param": other / (np.linalg.norm(other) + 1e-12) * norm}
            for method, vec in dirs.items():
                for sc in scales:
                    add = torch.tensor(sc * delta * vec[None, :], dtype=torch.float32, device=device)  # (B, d)

                    def edit(i, x, layer=li, add=add):
                        if i == layer:
                            x = x + add[:, None, :].to(x.dtype)
                        return x

                    d_model = implied(edit) - base
                    xo, ym = d_oracle.ravel(), d_model.ravel()
                    row = {"param": p, "layer": li, "method": method, "scale": float(sc),
                           "slope": float((xo @ ym) / max(xo @ xo, 1e-12)),
                           "mean_abs_model_change": float(np.abs(ym).mean()),
                           "mean_abs_oracle_change": float(np.abs(xo).mean())}
                    for p2 in PARAMS:
                        row[f"pattern_{p2}"] = _within_corr(patterns[p2], d_model)
                    row["pattern_own"] = row[f"pattern_{p}"]
                    row["pattern_best_other"] = max(row[f"pattern_{p2}"] for p2 in PARAMS if p2 != p)
                    rows.append(row)
    return rows
