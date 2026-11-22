"""How does the model forecast Hawkes processes, if not with exponential kernel heads?

The pilot showed excellent in-context forecasting (within ~6% of the oracle), linearly decodable
parameters and intensity, but no head with an exponential time profile that adapts to beta, and
an unused null slot. Three follow-ups:

M1  Behavioural intensity. The model's probability for the first (shortest) interval bin gives an
    implied intensity: lambda_hat = -log(1 - P(bin 0)) / width. Does it track the true lambda just
    after each event, and does it react to an event the way a Hawkes process should?

M2  Non-parametric attention spans. Per head, the attention-weighted mean time gap back to past
    events, compared across sequences with the true beta. A head whose span shrinks as beta grows
    adapts its memory to the kernel, whatever the shape of its profile (the pilot's exponential fit
    may simply have been the wrong shape).

M3  Where does long-range information flow? Restrict attention in one layer at a time to the last
    k events and measure the loss increase, split by beta tercile. A two-stage algorithm would show
    some layers needing long context (parameter estimation) and others only recent events (excitation).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from .task import make_batch


@torch.no_grad()
def implied_intensity(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, query_from: int = 32) -> dict:
    """M1: Spearman and R2 between log implied intensity and log true intensity, overall and by beta tercile."""
    model.eval()
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 88], edges, with_truth=True)
    logits = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device))["logits"]
    p0 = torch.softmax(logits.float(), -1)[..., 0].cpu().numpy()
    width = edges[1] - edges[0]
    lam_hat = -np.log1p(-np.clip(p0, 0, 1 - 1e-7)) / width
    q = np.arange(query_from, b["feats"].shape[1] - 1)
    lam = b["mu"][:, None] + b["E_after"][:, q]
    x, y = np.log(lam_hat[:, q] + 1e-9), np.log(lam)
    res = {"spearman": float(stats.spearmanr(x.ravel(), y.ravel()).correlation)}
    slope, icpt, r, _, _ = stats.linregress(y.ravel(), x.ravel())
    res.update({"r2": float(r ** 2), "slope": float(slope)})
    # within-sequence: does the implied intensity rise and fall with the excitation?
    within = [stats.spearmanr(x[s], y[s]).correlation for s in range(n_seqs) if np.ptp(y[s]) > 1e-6]
    res["within_seq_spearman_median"] = float(np.nanmedian(within))
    terc = np.quantile(b["beta"], [1 / 3, 2 / 3])
    for name, sel in (("beta_low", b["beta"] <= terc[0]), ("beta_mid", (b["beta"] > terc[0]) & (b["beta"] <= terc[1])),
                      ("beta_high", b["beta"] > terc[1])):
        res[f"within_seq_spearman_{name}"] = float(np.nanmedian([w for w, ok in zip(within, sel) if ok]))
    return res


@torch.no_grad()
def attention_spans(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, query_from: int = 32) -> list[dict]:
    """M2: per head, Spearman across sequences between attention-weighted mean time gap and true beta."""
    model.eval()
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 99], edges, with_truth=True)
    out = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device), return_attn=True)
    t = b["times64"]
    N = t.shape[1]
    q = np.arange(query_from, N - 1)
    gaps = t[:, q, None] - t[:, None, :]  # (B, Q, N); negative for future keys (masked anyway)
    lags = (q[:, None] - np.arange(N)[None, :]).astype(float)
    rows = []
    for li, A in enumerate(out["attn"]):
        A = A.float().cpu().numpy()[:, :, q, :N]  # drop the null slot
        for h in range(A.shape[1]):
            w = A[:, h]  # (B, Q, N)
            mass = w.sum(-1)
            wn = w / np.maximum(mass[..., None], 1e-9)
            span = (wn * np.clip(gaps, 0, None)).sum(-1).mean(1)  # per sequence
            lag = (wn * np.clip(lags, 0, None)[None]).sum(-1).mean(1)
            rho_t = stats.spearmanr(span, b["beta"]).correlation
            rho_lag = stats.spearmanr(lag, b["beta"]).correlation
            rows.append({"layer": li, "head": h, "median_span_time": float(np.median(span)), "median_span_events": float(np.median(lag)),
                         "spearman_span_vs_beta": float(rho_t), "spearman_lag_vs_beta": float(rho_lag),
                         "token_mass": float(mass.mean())})
    return rows


@torch.no_grad()
def window_ablation(model, dcfg: dict, edges: np.ndarray, n_seqs: int, seed, device, windows=(4, 16), query_from: int = 32,
                    batch: int = 256) -> list[dict]:
    """M3: loss increase (nats) at late queries when one layer may only read the last k events."""
    model.eval()
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 111], edges, with_truth=True)
    N = b["feats"].shape[1]
    terc = np.quantile(b["beta"], [1 / 3, 2 / 3])
    grp = np.digitize(b["beta"], terc)  # 0 low, 1 mid, 2 high beta

    def loss(mask_layer=None, k=None):
        tot = []
        for i in range(0, n_seqs, batch):
            sl = slice(i, i + batch)
            feats = torch.from_numpy(b["feats"][sl]).to(device)
            pos = torch.from_numpy(b["times"][sl]).to(device)
            am = None
            if mask_layer is not None:
                ii = torch.arange(N, device=device)
                band = (ii[:, None] - ii[None, :]) < k
                am = {mask_layer: band[None].expand(feats.shape[0], N, N)}
            lg = model(feats=feats, pos=pos, attn_mask=am)["logits"].float()
            y = torch.from_numpy(b["labels"][sl]).to(device)
            ce = F.cross_entropy(lg.transpose(1, 2), y, ignore_index=-100, reduction="none")
            tot.append(ce[:, query_from:N - 1].mean(1).cpu().numpy())
        return np.concatenate(tot)

    base = loss()
    rows = []
    for li in range(len(model.blocks)):
        for k in windows:
            d = loss(li, k) - base
            row = {"layer": li, "window": int(k), "delta_nats": float(d.mean())}
            for g, name in enumerate(["beta_low", "beta_mid", "beta_high"]):
                row[f"delta_{name}"] = float(d[grp == g].mean())
            rows.append(row)
    return rows
