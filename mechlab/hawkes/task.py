"""In-context Hawkes estimation: does attention compute the excitation kernel?

The model reads a sequence of event times from a Hawkes process with unknown (mu, alpha, beta)
and predicts a histogram over the next inter-event time. Attention uses *time-rotary* embeddings,
so each head's attention logit depends on content and on the time gap t_i - t_j only. A learned
"null slot" lets a head put weight on nothing, so it can compute un-normalised sums such as
sum_j exp(-beta (t_i - t_j)), which is exactly the Hawkes excitation.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from ..core.gpt import GPTConfig, TinyGPT
from ..core.probes import ridge_cv
from .simulate import (bin_edges, oracle_bin_probs, oracle_bin_probs_powerlaw, poisson_bin_probs, sample_params,
                       sample_params_powerlaw, simulate, simulate_powerlaw, to_bins)

N_FEATURES = 2


def features(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Per-event inputs: scaled log interval before the event, scaled log elapsed time."""
    f1 = (np.log(np.maximum(intervals, 1e-6)) + 1.0) / 2.5
    f2 = np.log1p(times) / 3.0
    return np.stack([f1, f2], -1).astype(np.float32)


def kernel_of(dcfg: dict) -> str:
    return str(dcfg.get("kernel", "exp"))


def make_batch(dcfg: dict, seed, edges: np.ndarray, with_truth: bool = False, param_ranges: dict | None = None) -> dict:
    """kernel "exp" (default): parameters mu, alpha, beta. kernel "powerlaw": mu, alpha, c, gamma; for code
    shared with the exponential case, ``beta`` is set to 1/c (the kernel's rate scale)."""
    rng = np.random.default_rng(seed)
    B, N = int(dcfg["batch_size"]), int(dcfg["n_events"])
    pr = {**dcfg.get("params", {}), **(param_ranges or {})}
    if kernel_of(dcfg) == "powerlaw":
        p = sample_params_powerlaw(rng, B, mu=tuple(pr.get("mu", (0.2, 2.0))), alpha=tuple(pr.get("alpha", (0.0, 0.9))),
                                   c=tuple(pr.get("c", (0.1, 2.0))), gamma=tuple(pr.get("gamma", (1.5, 3.0))))
        s = simulate_powerlaw(p["mu"], p["alpha"], p["c"], p["gamma"], N, rng)
        p["beta"] = 1.0 / p["c"]
    else:
        p = sample_params(rng, B, mu=tuple(pr.get("mu", (0.2, 2.0))), alpha=tuple(pr.get("alpha", (0.0, 0.9))),
                          beta=tuple(pr.get("beta", (0.5, 10.0))))
        s = simulate(p["mu"], p["alpha"], p["beta"], N, rng)
    y = np.full((B, N), -100, dtype=np.int64)
    y[:, :-1] = to_bins(s["intervals"][:, 1:], edges)  # token i predicts the interval to event i+1
    out = {"feats": features(s["times"], s["intervals"]), "times": s["times"].astype(np.float32), "labels": y}
    if with_truth:
        out.update({**p, "E_after": s["E_after"], "times64": s["times"], "intervals": s["intervals"]})
    return out


def oracle_probs(dcfg: dict, b: dict, edges: np.ndarray) -> np.ndarray:
    """Exact next-interval bin probabilities after every event except the last, (B, N - 1, n_bins)."""
    if kernel_of(dcfg) == "powerlaw":
        return oracle_bin_probs_powerlaw(b["times64"], b["mu"], b["alpha"], b["c"], b["gamma"], edges)[:, :-1]
    return oracle_bin_probs(b["mu"][:, None], b["E_after"][:, :-1], b["beta"][:, None], edges)


def batch_for_step(step: int, dcfg: dict, seed: int, edges: np.ndarray) -> dict:
    return make_batch(dcfg, [seed, step], edges)


def make_batch_fn(dcfg: dict, seed: int, edges: np.ndarray):
    return partial(batch_for_step, dcfg=dcfg, seed=seed, edges=edges)


def build_model(mcfg: dict, n_bins: int) -> TinyGPT:
    hd = mcfg["d_model"] // mcfg["n_heads"]
    fr = mcfg.get("time_freqs", [0.02, 50.0])
    freqs = np.exp(np.linspace(np.log(fr[0]), np.log(fr[1]), hd // 2)).tolist()
    cfg = GPTConfig.from_dict({**mcfg, "n_features": N_FEATURES, "vocab_size": 0, "n_out": n_bins, "pos": mcfg.get("pos", "rope"),
                               "rope_freqs": freqs, "null_attn": bool(mcfg.get("null_attn", True))})
    return TinyGPT(cfg)


def loss_fn(model, batch):
    out = model(feats=batch["feats"], pos=batch["times"])
    logits = out["logits"].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100)
    return loss, {}


@torch.no_grad()
def evaluate(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, param_ranges=None, batch: int = 256) -> dict:
    """Cross-entropy (nats) of model, oracle (true parameters) and in-context Poisson, by context length."""
    model.eval()
    N = int(dcfg["n_events"])
    mid = min(64, (N - 1) // 2)
    buckets = [b for b in [(0, 8), (8, 32), (32, mid), (mid, N - 1)] if b[1] > b[0]]
    acc = {k: [] for k in ("model", "oracle", "poisson")}
    for i in range(0, n_seqs, batch):
        d = {**dcfg, "batch_size": min(batch, n_seqs - i)}
        b = make_batch(d, [seed, i, 31], edges, with_truth=True, param_ranges=param_ranges)
        logits = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device))["logits"]
        logp = torch.log_softmax(logits.float(), -1).cpu().numpy()[:, :-1]
        y = b["labels"][:, :-1]
        ce_m = -np.take_along_axis(logp, y[..., None], -1)[..., 0]
        P = oracle_probs(dcfg, b, edges)
        ce_o = -np.log(np.take_along_axis(P, y[..., None], -1)[..., 0])
        rate = np.arange(1, N)[None, :] / b["times64"][:, :-1]
        Pp = poisson_bin_probs(rate, edges)
        ce_p = -np.log(np.take_along_axis(Pp, y[..., None], -1)[..., 0])
        acc["model"].append(ce_m); acc["oracle"].append(ce_o); acc["poisson"].append(ce_p)
    ce = {k: np.concatenate(v) for k, v in acc.items()}
    res = {}
    for lo, hi in buckets:
        sl = slice(lo, hi)
        m, o, p = ce["model"][:, sl].mean(), ce["oracle"][:, sl].mean(), ce["poisson"][:, sl].mean()
        key = f"ctx{lo}-{hi}"
        res[f"{key}_model"], res[f"{key}_oracle"], res[f"{key}_poisson"] = float(m), float(o), float(p)
        res[f"{key}_gap_frac"] = float((m - o) / max(p - o, 1e-9))  # 0 = oracle, 1 = no better than Poisson
    return res


@torch.no_grad()
def collect(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, query_from: int = 32) -> dict:
    """Residuals and attention at query events i >= query_from, with the true parameters and intensity."""
    model.eval()
    d = {**dcfg, "batch_size": n_seqs}
    b = make_batch(d, [seed, 77], edges, with_truth=True)
    out = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device),
                return_resid=True, return_attn=True)
    N = b["feats"].shape[1]
    q = np.arange(query_from, N - 1)
    resid = [r.float().cpu().numpy()[:, q].reshape(-1, r.shape[-1]) for r in out["resid"]]
    lam = b["mu"][:, None] + b["E_after"][:, q]
    targets = {
        "log_mu": np.repeat(np.log(b["mu"]), len(q)),
        "alpha": np.repeat(b["alpha"], len(q)),
        "log_beta": np.repeat(np.log(b["beta"]), len(q)),  # power-law kernel: log(1/c), the rate scale
    }
    if "gamma" in b:
        targets["gamma"] = np.repeat(b["gamma"], len(q))
    targets.update({"log_intensity": np.log(lam).reshape(-1), "log_excitation": np.log(b["E_after"][:, q] + 1e-9).reshape(-1)})
    attn = [a.float().cpu().numpy() for a in out["attn"]]  # (B, H, N, N+1)
    return {"resid": resid, "targets": targets, "groups": np.repeat(np.arange(n_seqs), len(q)), "attn": attn,
            "times": b["times64"], "mu": b["mu"], "alpha": b["alpha"], "beta": b["beta"], "query_idx": q}


def probe_params(col: dict) -> list[dict]:
    rows = []
    for li, X in enumerate(col["resid"]):
        row = {"layer": li}
        for name, y in col["targets"].items():
            r = ridge_cv(X, y, groups=col["groups"], alphas=(0.1, 1, 10, 100))
            row[name] = float(r["r2"][0])
        rows.append(row)
    return rows


def kernel_fits(col: dict, max_queries: int = 24, min_w: float = 1e-8) -> list[dict]:
    """For each head, fit log(w_ij / w_null) = a - b * (t_i - t_j) over past events j at query events i.

    Reports the median fit R^2 (how exponential the head's time profile is), the median decay b,
    its spread across sequences, and the Spearman correlation of b with the true beta (does the head
    *adapt* its decay to the sequence, i.e. estimate the kernel in context?). Heads with high R^2 but
    no correlation are fixed exponential filters (a candidate "filter bank").

    The same weights are also fitted with a power-law profile, log w = a - g * log(1 + gap / s), where s is
    the sequence's mean interval so far (a fixed, data-defined scale, so both fits have two free
    parameters). median_r2_pow > median_r2 means the head's memory looks long-tailed rather than exponential.
    """
    rows = []
    B = col["times"].shape[0]
    rng = np.random.default_rng(0)
    qs = col["query_idx"]
    for li, A in enumerate(col["attn"]):
        has_null = A.shape[-1] == A.shape[-2] + 1
        H = A.shape[1]
        for h in range(H):
            slopes, r2s, betas, r2p = [], [], [], []
            for s in range(B):
                for i in rng.choice(qs, size=min(max_queries, len(qs)), replace=False):
                    w = A[s, h, i, : i + 1]
                    dt = col["times"][s, i] - col["times"][s, : i + 1]
                    ok = w > min_w
                    if ok.sum() < 6:
                        continue
                    yv = np.log(w[ok])  # the reference (null slot or not) only shifts the intercept
                    xv = dt[ok]
                    if np.ptp(xv) <= 0:
                        continue
                    slope, icpt, r, _, _ = stats.linregress(xv, yv)
                    slopes.append(-slope)
                    r2s.append(r ** 2)
                    betas.append(col["beta"][s])
                    scale = max(col["times"][s, i] / max(i, 1), 1e-6)
                    r2p.append(stats.linregress(np.log1p(xv / scale), yv).rvalue ** 2)
            if not slopes:
                continue
            slopes, r2s, betas = map(np.asarray, (slopes, r2s, betas))
            rho = stats.spearmanr(slopes, betas).correlation if len(slopes) > 10 else np.nan
            null_mass = float(A[:, h, qs, -1].mean()) if has_null else float("nan")
            rows.append({"layer": li, "head": h, "median_r2": float(np.median(r2s)), "median_r2_pow": float(np.median(r2p)),
                         "median_decay": float(np.median(slopes)),
                         "decay_iqr": float(np.subtract(*np.percentile(slopes, [75, 25]))),
                         "spearman_decay_vs_beta": float(rho), "null_mass": null_mass, "n_fits": int(len(slopes))})
    return rows
