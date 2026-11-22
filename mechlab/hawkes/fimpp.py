"""Exploratory: does a published point-process foundation model forecast like the Bayes posterior predictive?

FIM-PP (Berghaus et al., ICLR 2026; github.com/FIM4Science/OpenFIM, MIT licence) reads a *context set* of event
sequences from one process plus the history of an inference sequence, and outputs, after every inference event i,
a local intensity mu_i + (alpha_i - mu_i) exp(-beta_i (t - t_i)) on a normalised time axis (times divided by the
largest inter-event time in the context set, c). On the original axis the compensator of the next interval tau is
mu_i x + (alpha_i - mu_i)(1 - exp(-beta_i x)) / beta_i with x = tau / c, which gives exact next-interval bin
probabilities, directly comparable with our transformers and with Bayes.

For an exponential-kernel Hawkes process, mu_i / c estimates the baseline mu, beta_i / c the decay beta and
alpha_i / c the intensity just after event i. We compare those estimates with the posterior means given the same
data (importance sampling over our prior) and with the truth, as for our own models.

The audit uses OUR prior (FIM-PP was trained on a different, wider family), so "Bayes" is the best forecaster for the
data shown, not for FIM-PP's own training prior. With many context paths the posterior concentrates and Bayes
approaches the oracle; Bayes is computed only up to ``max_k_bayes`` context paths.
"""
from __future__ import annotations

import numpy as np
import torch

from .baselines import _loglik_checkpoints
from .posterior import loglik_exp, prior_draws
from .simulate import oracle_bin_probs, sample_params, simulate, to_bins


def load_pretrained(device, repo_ids=("FIM4Science/fim-pp", "FIM4Science/FIM-PP")):
    """Download and load FIM-PP. Needs internet, huggingface_hub and the OpenFIM source on sys.path."""
    from huggingface_hub import snapshot_download

    last = None
    for rid in repo_ids:
        try:
            path = snapshot_download(repo_id=rid, repo_type="model")
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(path, trust_remote_code=True)
            except Exception as e:  # pragma: no cover - fall back to the package class
                last = e
                from fim.models.hawkes import FIMHawkes
                model = FIMHawkes.from_pretrained(path)
            return model.to(device).eval(), rid
        except Exception as e:  # pragma: no cover - network / API variations
            last = e
    raise RuntimeError(f"could not load FIM-PP: {last}")


def random_model(config_yaml: str, device):
    """FIM-PP architecture with random weights, from an OpenFIM training config (offline tests)."""
    import yaml
    from fim.models.hawkes import FIMHawkes, FIMHawkesConfig

    cfg = yaml.load(open(config_yaml), Loader=yaml.UnsafeLoader)
    return FIMHawkes(FIMHawkesConfig(**dict(cfg["model"]))).to(device).eval()


def simulate_sets(n_proc: int, n_paths: int, n_events: int, prior: dict, seed) -> dict:
    """n_proc processes from our exponential prior, each with n_paths sequences of n_events events."""
    rng = np.random.default_rng(seed)
    p = sample_params(rng, n_proc, mu=tuple(prior.get("mu", (0.2, 2.0))), alpha=tuple(prior.get("alpha", (0.0, 0.9))),
                      beta=tuple(prior.get("beta", (0.5, 10.0))))
    rep = {k: np.repeat(v, n_paths) for k, v in p.items()}
    s = simulate(rep["mu"], rep["alpha"], rep["beta"], n_events, rng)
    return {**p, "times": s["times"].reshape(n_proc, n_paths, n_events),
            "E_after": s["E_after"].reshape(n_proc, n_paths, n_events)}


def _inputs(ctx: np.ndarray, inf: np.ndarray, device) -> dict:
    B, P, L = ctx.shape
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return {"context_event_times": t(ctx)[..., None], "context_event_types": torch.zeros(B, P, L, 1, dtype=torch.long, device=device),
            "context_seq_lengths": torch.full((B, P), L, device=device),
            "inference_event_times": t(inf)[:, None, :, None],
            "inference_event_types": torch.zeros(B, 1, inf.shape[1], 1, dtype=torch.long, device=device),
            "inference_seq_lengths": torch.full((B, 1), inf.shape[1], device=device),
            "intensity_evaluation_times": torch.zeros(B, 1, 1, device=device), "num_marks": 1}


@torch.no_grad()
def forecast(model, ctx: np.ndarray, inf: np.ndarray, edges: np.ndarray, device, batch: int = 8) -> dict:
    """Next-interval bin probabilities after every inference event (B, L, bins), the local parameters on the original
    time scale, and the decoder states (B, L, D)."""
    hid = []
    h = model.ts_decoder.register_forward_hook(lambda m, i, o: hid.append(o.detach().float().cpu()))
    probs, pars = [], {k: [] for k in ("mu", "alpha", "beta")}
    ef = np.asarray(edges[1:-1])
    try:
        for a in range(0, len(ctx), batch):
            out = model(_inputs(ctx[a:a + batch], inf[a:a + batch], device))
            f = out["intensity_function"]
            c = f.norm_constants.float().cpu().numpy()[:, None, None]                       # (b,1,1)
            mu, al, be = (getattr(f, k)[:, 0, 0].float().cpu().numpy()[..., None] for k in ("mu", "alpha", "beta"))  # (b,L,1)
            x = ef[None, None, :] / c                                                       # (b,1,E)
            comp = mu * x + (al - mu) * (1 - np.exp(-be * x)) / np.maximum(be, 1e-8)
            S = np.concatenate([np.ones(comp.shape[:2] + (1,)), np.exp(-comp), np.zeros(comp.shape[:2] + (1,))], -1)
            P = np.clip(S[..., :-1] - S[..., 1:], 1e-12, None)
            probs.append(P / P.sum(-1, keepdims=True))
            for k, v in (("mu", mu), ("alpha", al), ("beta", be)):
                pars[k].append(v[..., 0] / c[..., 0])
    finally:
        h.remove()
    H = torch.cat(hid).numpy() if hid else None
    return {"P": np.concatenate(probs), **{k: np.concatenate(v) for k, v in pars.items()},
            "hidden": H.reshape(len(ctx), inf.shape[1], -1) if H is not None else None}


def bayes_with_context(ctx: np.ndarray, inf: np.ndarray, checkpoints, edges, prior: dict) -> dict:
    """Posterior predictive given all context paths and the first m inference events (one process)."""
    base = np.zeros(len(prior["mu"]))
    for path in ctx:
        base += loglik_exp(path, prior, len(path))[:, -1]
    lld, Ed = _loglik_checkpoints(inf, prior["mu"], prior["alpha"], prior["beta"], checkpoints)
    out = {"P": [], "ess": [], "post": {k: [] for k in ("log_mu", "alpha", "log_beta")}}
    theta = {"log_mu": np.log(prior["mu"]), "alpha": prior["alpha"], "log_beta": np.log(prior["beta"])}
    for m in checkpoints:
        ll = base + lld[m]
        w = np.exp(ll - ll.max())
        w /= w.sum()
        out["ess"].append(1.0 / np.sum(w ** 2))
        keep = w > 1e-10 * w.max()
        out["P"].append((w[keep] / w[keep].sum()) @ oracle_bin_probs(prior["mu"][keep], Ed[m][keep], prior["beta"][keep], edges))
        for k, v in theta.items():
            out["post"][k].append(float(v @ w))
    return {"P": np.array(out["P"]), "ess": np.array(out["ess"]), "post": {k: np.array(v) for k, v in out["post"].items()}}


def audit(model, edges: np.ndarray, device, prior_cfg: dict, Ks=(1, 4, 16, 64, 256), n_proc: int = 200,
          n_events: int = 100, checkpoints=(2, 4, 8, 16, 32, 64, 96), n_prior: int = 20000, max_k_bayes: int = 16,
          seed: int = 0, own_model=None) -> dict:
    """Cross-entropy of FIM-PP, Bayes (with the same context) and the oracle by number of context paths K, and how its
    parameter estimates relate to the posterior means and the truth."""
    from ..core.probes import r2_per_output

    cps = [m for m in checkpoints if m < n_events]
    idx = np.array(cps) - 1
    prior = prior_draws({"params": prior_cfg}, n_prior, seed + 99)
    res = {"Ks": list(Ks), "checkpoints": cps, "rows": []}
    for K in Ks:
        d = simulate_sets(n_proc, K + 1, n_events, prior_cfg, [seed, K])
        ctx, inf = d["times"][:, :K], d["times"][:, K]
        fc = forecast(model, ctx, inf, edges, device, batch=max(1, 4096 // max(K, 1) // 8))
        y = to_bins(np.diff(inf, axis=1), edges)                                              # (n, N-1): interval after event i
        yk = y[:, idx]
        q = fc["P"][:, idx]
        orc = oracle_bin_probs(d["mu"][:, None], d["E_after"][:, K][:, idx], d["beta"][:, None], edges)
        ce = lambda Pm: float(-np.log(np.clip(np.take_along_axis(Pm, yk[..., None], -1)[..., 0], 1e-300, None)).mean())  # noqa: E731
        row = {"K": int(K), "ce_fimpp": ce(q), "ce_oracle": ce(orc),
               "by_context": [{"m": int(m), "ce_fimpp": ce(q[:, [c]]), "ce_oracle": ce(orc[:, [c]])} for c, m in enumerate(cps)]}
        est = {"log_mu": np.log(np.maximum(fc["mu"][:, idx], 1e-9)), "log_beta": np.log(np.maximum(fc["beta"][:, idx], 1e-9))}
        truth = {"log_mu": np.repeat(np.log(d["mu"])[:, None], len(idx), 1), "log_beta": np.repeat(np.log(d["beta"])[:, None], len(idx), 1)}
        r2 = lambda a, b: float(r2_per_output(a.reshape(-1, 1), b.reshape(-1, 1))[0])  # noqa: E731
        corr = lambda a, b: float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])  # noqa: E731
        for k in est:
            row[f"{k}_corr_est_true"] = corr(est[k], truth[k])
        if K <= max_k_bayes:
            bys = [bayes_with_context(ctx[i], inf[i], cps, edges, prior) for i in range(n_proc)]
            Pb = np.stack([b["P"] for b in bys])
            row["ce_bayes"] = ce(Pb)
            row["kl_bayes_fimpp"] = float((Pb * (np.log(np.clip(Pb, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum(-1).mean())
            row["median_ess"] = float(np.median([b["ess"] for b in bys]))
            for c, m in enumerate(cps):
                row["by_context"][c]["ce_bayes"] = ce(Pb[:, [c]])
            for k in est:
                pm = np.stack([b["post"][k] for b in bys])
                row[f"{k}_corr_est_posterior"] = corr(est[k], pm)
                row[f"{k}_r2_posterior_vs_true"] = r2(truth[k], pm)
                if fc["hidden"] is not None:
                    from ..core.probes import ridge_cv
                    X = fc["hidden"][:, idx].reshape(-1, fc["hidden"].shape[-1])
                    g = np.repeat(np.arange(n_proc), len(idx))
                    row[f"{k}_probe_r2_posterior"] = float(ridge_cv(X, pm.reshape(-1), groups=g, alphas=(1, 10, 100))["r2"][0])
                    row[f"{k}_probe_r2_true"] = float(ridge_cv(X, truth[k].reshape(-1), groups=g, alphas=(1, 10, 100))["r2"][0])
        if own_model is not None and K == min(Ks):
            from .task import features
            iv = np.diff(np.concatenate([np.zeros((n_proc, 1)), inf], 1), axis=1)
            with torch.no_grad():
                lg = own_model(feats=torch.from_numpy(features(inf, iv)).to(device),
                               pos=torch.as_tensor(inf, dtype=torch.float32, device=device))["logits"]
            Po = torch.softmax(lg.float(), -1).cpu().numpy()[:, idx]
            row["ce_own_transformer_no_context"] = ce(Po)
            b0 = [bayes_with_context(np.zeros((0, n_events)), inf[i], cps, edges, prior) for i in range(n_proc)]
            row["ce_bayes_no_context"] = ce(np.stack([b["P"] for b in b0]))
        res["rows"].append(row)
        print(f"[fimpp] K={K}: CE FIM-PP {row['ce_fimpp']:.3f} | Bayes {row.get('ce_bayes', float('nan')):.3f} | "
              f"oracle {row['ce_oracle']:.3f}", flush=True)
    return res
