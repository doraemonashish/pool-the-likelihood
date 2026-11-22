"""Round 8 for the Hawkes paper: how the posterior is computed, and whether it is a causal bottleneck.

Round 7 showed (M1) that every model forecasts like the Bayes predictive of its own training prior even on data from
another kernel family, that (K1) decayed event counts at the current time are NOT sufficient for the posterior, and
that (K2) block 0's output is mostly a linear function of them. The exponential-Hawkes log-likelihood at decay b is

    sum_j log(mu + alpha b R_j(b)) - mu t - alpha (n - S_b(t)),   R_j(b) = sum_{k<j} exp(-b (t_j - t_k)),

a sum over past events of a nonlinear function of each event's decayed count. So the posterior is a function of
POOLED traces, mean_j log(c + R_j(b)) for a few offsets c, which is what attention with uniform weights over past
events computes from per-event traces. A CPU check found these pooled traces explain the posterior means with
R^2 0.99 / 0.99 / 0.94 (log mu, alpha, log beta). Tests here:

P1  sufficiency of pooled traces (dense rates and offsets), against the best linear probe on the residual stream
P2  representation: pooled traces are linearly decodable from the residual stream, more after block 0 than before
P3  attention: heads in blocks >= 1 that spread their weight evenly over past events (pooling heads)
P4  reconstruction: replacing a block's output by a linear map of [bank, pooled] features recovers more of the loss
    than [bank] alone
I2  posterior interchange done properly: every block, patching the last position only or the whole prefix, with
    decoder (probe) directions, encoder directions, and a subspace learned by distributed alignment search (DAS)
M1  confidence intervals for the misspecification result (per-sequence bootstrap) and importance-sampling ESS
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.probes import mlp_probe, ridge_cv, ridge_fit
from . import r7
from .posterior import posterior_stats
from .task import collect, kernel_of, make_batch

DENSE_RATES = tuple(float(x) for x in np.geomspace(0.25, 20.0, 12))
DENSE_CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
PARAMS = r7.PARAMS


def trace_features(times: np.ndarray, feats: np.ndarray, rates=DENSE_RATES, cs=DENSE_CS) -> dict:
    """Per position (causal): bank (current decayed counts, 2R), pooled (mean over events j <= i of log(c + R_j(b)),
    R x len(cs)), basic (inputs, log count, log time)."""
    fb = r7.bank_features(times, feats, rates)
    R = len(rates)
    Rj = np.expm1(fb["bank"][..., :R]) - 1.0                      # decayed count of EARLIER events at each event
    Rj = np.maximum(Rj, 0.0)
    N = times.shape[1]
    denom = np.arange(1, N + 1)[None, :, None]
    pooled = np.concatenate([np.cumsum(np.log(c + Rj), 1) / denom for c in cs], -1)
    return {"bank": fb["bank"].astype(np.float32), "pooled": pooled.astype(np.float32),
            "basic": fb["basic"].astype(np.float32), "rates": list(rates), "cs": list(cs)}


def _std(X):
    X = X.reshape(-1, X.shape[-1]).astype(np.float64)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


# ------------------------------------------------------------------------------------------- P1-P4 pooled traces
def attention_pooling(attn: list, q: np.ndarray, has_null: bool, every: int = 4) -> list[dict]:
    """Per block and head: mean total-variation distance between the attention over past events (renormalised,
    null slot excluded) and the uniform distribution over them; mean null mass; mean relative age of attended events
    (0 = newest, 1 = oldest; uniform = 0.5)."""
    rows = []
    qs = q[::every]
    for bi, A in enumerate(attn):                         # (B, H, N, N[+1])
        B, H = A.shape[:2]
        for h in range(H):
            tv, null, age = [], [], []
            for i in qs:
                w = A[:, h, i, :i + 1].astype(np.float64)
                tot = w.sum(-1, keepdims=True)
                e = w / np.maximum(tot, 1e-12)
                tv.append(0.5 * np.abs(e - 1.0 / (i + 1)).sum(-1).mean())
                null.append(float(A[:, h, i, -1].mean()) if has_null else 0.0)
                age.append(float((e * ((i - np.arange(i + 1)) / max(i, 1))[None]).sum(-1).mean()))
            rows.append({"block": bi, "head": h, "tv_uniform": float(np.mean(tv)), "null_mass": float(np.mean(null)),
                         "mean_age": float(np.mean(age))})
    return rows


def pooled_tests(model, dcfg: dict, edges, n_seqs: int, seed, device, K: int = 20000, n_test: int = 400,
                 query_from: int = 48, min_damage: float = 0.01, mlp_epochs: int = 200) -> dict:
    assert kernel_of(dcfg) == "exp"
    model.eval()
    col = collect(model, dcfg, edges, n_seqs, seed, device, query_from=query_from)
    q = col["query_idx"]
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 77], edges, with_truth=True)      # the batch collect used
    tf = trace_features(b["times64"], b["feats"])
    post = posterior_stats(dcfg, col["times"], q, K=K, seed=seed + 1, device="cuda" if torch.cuda.is_available() else "cpu")
    groups = col["groups"]
    mdev = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {"count_time": [tf["basic"]], "bank": [tf["bank"], tf["basic"]], "pooled": [tf["pooled"], tf["basic"]],
            "bank_pooled": [tf["bank"], tf["pooled"], tf["basic"]]}
    X = {k: _std(np.concatenate([a[:, q] for a in v], -1)) for k, v in sets.items()}
    res = {"rates": tf["rates"], "cs": tf["cs"], "sufficiency": {}, "representation": [], "attention": [],
           "reconstruction": []}
    for k in PARAMS:
        y = post[k]["mean"].reshape(-1)
        row = {f"{name}_mlp": float(mlp_probe(Xs, y, groups=groups, epochs=mlp_epochs, device=mdev)["r2"][0])
               for name, Xs in X.items()}
        row["residual_best_linear"] = max(float(ridge_cv(R, y, groups=groups, alphas=(0.1, 1, 10, 100))["r2"][0])
                                          for R in col["resid"])
        res["sufficiency"][k] = row
    # P2: how well each layer's residual linearly encodes the pooled and the current traces
    Yp = tf["pooled"][:, q].reshape(-1, tf["pooled"].shape[-1])
    Yb = tf["bank"][:, q, :len(tf["rates"])].reshape(-1, len(tf["rates"]))
    for li, R in enumerate(col["resid"]):
        rp = ridge_cv(R, Yp, groups=groups, alphas=(0.1, 1, 10, 100))["r2"]
        rb = ridge_cv(R, Yb, groups=groups, alphas=(0.1, 1, 10, 100))["r2"]
        res["representation"].append({"layer": li, "pooled_r2_mean": float(np.nanmean(rp)),
                                      "pooled_r2_median": float(np.nanmedian(rp)), "bank_r2_mean": float(np.nanmean(rb))})
    # P3: attention pooling
    has_null = col["attn"][0].shape[-1] == col["attn"][0].shape[-2] + 1
    res["attention"] = attention_pooling(col["attn"], q, has_null)
    del col
    # P4: reconstruction with and without pooled traces
    test = make_batch({**dcfg, "batch_size": n_test}, [seed, 78], edges, with_truth=True)
    fit_b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 79], edges, with_truth=True)
    ft, ff = trace_features(test["times64"], test["feats"]), trace_features(fit_b["times64"], fit_b["feats"])
    with torch.no_grad():
        full_fit = model(feats=torch.from_numpy(fit_b["feats"]).to(device), pos=torch.from_numpy(fit_b["times"]).to(device),
                         return_resid=True)["resid"]
    y = test["labels"]
    late = np.zeros_like(y, dtype=bool)
    late[:, 32:-1] = True

    def ce(logits):
        lp = torch.log_softmax(logits.detach().float(), -1).cpu().numpy()
        return float(-np.take_along_axis(lp, np.maximum(y, 0)[..., None], -1)[..., 0][late].mean())

    def fwd(edit=None):
        with torch.no_grad():
            return model(feats=torch.from_numpy(test["feats"]).to(device), pos=torch.from_numpy(test["times"]).to(device),
                         edit=edit)["logits"]

    base = ce(fwd())
    feats_sets = {"count_time": ("basic",), "bank": ("bank", "basic"), "bank_pooled": ("bank", "pooled", "basic")}
    for li in range(len(full_fit) - 2):
        Hf = full_fit[li + 1].float().cpu().numpy().reshape(-1, full_fit[li + 1].shape[-1])
        row = {"block": li, "ce_original": base}
        for name, keys in feats_sets.items():
            Ff = np.concatenate([ff[k] for k in keys], -1)
            Ft = np.concatenate([ft[k] for k in keys], -1)
            Wr = ridge_fit(Ff.reshape(-1, Ff.shape[-1]), Hf, alpha=1.0)
            Hhat = torch.as_tensor(Ft.reshape(-1, Ft.shape[-1]) @ Wr["W"] + Wr["b"], dtype=torch.float32,
                                   device=device).reshape(n_test, Ft.shape[1], -1)

            def hook(i, x, li=li, Hhat=Hhat):
                return Hhat.to(x.dtype) if i == li else x

            row[f"ce_{name}"] = ce(fwd(hook))
        dmg = row["ce_count_time"] - base
        row["damage_count_time"] = float(dmg)
        for name in ("bank", "bank_pooled"):
            row[f"recovered_{name}"] = float((row["ce_count_time"] - row[f"ce_{name}"]) / dmg) if dmg >= min_damage else None
        res["reconstruction"].append(row)
    return res


# ------------------------------------------------------------------------------------------- I2 interchange + DAS
def _orth(W: np.ndarray) -> np.ndarray:
    return np.linalg.qr(W)[0]


def _bayes_transplant(dcfg, A, B, checkpoints, edges, K, seed, device):
    bayesA = r7.bayes_dists(dcfg, A["times64"], checkpoints, edges, K, seed, device, return_draws=True)
    prior = bayesA["prior"]
    n = len(B["times64"])
    BT = np.zeros((n, len(checkpoints), len(edges) - 1))
    BB = np.zeros_like(BT)
    for s in range(n):
        lld, Ed = r7._loglik_checkpoints(B["times64"][s], prior["mu"], prior["alpha"], prior["beta"], checkpoints)
        for c, m in enumerate(checkpoints):
            for target, ll in ((BT, bayesA["lls"][s, :, c]), (BB, lld[m])):
                w = np.exp(ll - ll.max())
                w /= w.sum()
                keep = w > 1e-10 * w.max()
                Pk = r7.oracle_bin_probs(prior["mu"][keep], Ed[m][keep], prior["beta"][keep], edges)
                target[s, c] = (w[keep] / w[keep].sum()) @ Pk
    return BT, BB


def interchange_das(model, dcfg: dict, edges, n_fit: int, n_pairs: int, seed, device, checkpoints=(64, 128),
                    K: int = 20000, ks=(3, 8), steps: int = 300, lr: float = 0.02, modes=("query", "prefix"),
                    query_from: int = 48) -> dict:
    """Patch B's residual after block l toward A's inside a subspace; transfer = 1 - KL(BT || patched) / KL(BT || B),
    where BT is the Bayes forecast with A's posterior weights on B's history. DAS subspaces are trained on the first
    half of the pairs and every method is evaluated on the second half."""
    assert kernel_of(dcfg) == "exp"
    model.eval()
    col = collect(model, dcfg, edges, n_fit, seed, device, query_from=query_from)
    post = posterior_stats(dcfg, col["times"], col["query_idx"], K=K, seed=seed + 1,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    Y = np.stack([post[k]["mean"].reshape(-1) for k in PARAMS], 1)
    Ys = (Y - Y.mean(0)) / Y.std(0)
    A = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 601], edges, with_truth=True)
    B = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 602], edges, with_truth=True)
    BT, BB = _bayes_transplant(dcfg, A, B, checkpoints, edges, K, seed + 3, device)
    with torch.no_grad():
        residA = [r.float() for r in model(feats=torch.from_numpy(A["feats"]).to(device),
                                           pos=torch.from_numpy(A["times"]).to(device), return_resid=True)["resid"]]
        baseB = torch.log_softmax(model(feats=torch.from_numpy(B["feats"]).to(device),
                                        pos=torch.from_numpy(B["times"]).to(device))["logits"].float(), -1)
    fB, tB = torch.from_numpy(B["feats"]).to(device), torch.from_numpy(B["times"]).to(device)
    half = n_pairs // 2
    tr, te = slice(0, half), slice(half, n_pairs)
    BT_t = torch.as_tensor(BT, dtype=torch.float32, device=device).clamp_min(1e-12)
    d = residA[0].shape[-1]
    n_blocks = len(residA) - 1
    rng = np.random.default_rng(seed + 9)
    for p in model.parameters():
        p.requires_grad_(False)

    def patched_lp(sl, li, Q, m, mode):
        xa = residA[li + 1][sl]
        pos = slice(m - 1, m) if mode == "query" else slice(0, m)

        def hook(i, x):
            if i == li:
                x = x.clone()
                diff = xa[:, pos].to(x.dtype) - x[:, pos]
                x[:, pos] = x[:, pos] + (diff @ Q.to(x.dtype)) @ Q.to(x.dtype).T
            return x

        out = model(feats=fB[sl], pos=tB[sl], edit=hook)
        return torch.log_softmax(out["logits"][:, m - 1].float(), -1)

    def kl_to_bt(sl, lp, c):
        P = BT_t[sl, c]
        return (P * (P.log() - lp)).sum(-1)

    def transfer(sl, Q, li, mode):
        vals = []
        with torch.no_grad():
            for c, m in enumerate(checkpoints):
                kp = kl_to_bt(sl, patched_lp(sl, li, Q, m, mode), c).mean().item()
                k0 = kl_to_bt(sl, baseB[sl, m - 1], c).mean().item()
                vals.append(1 - kp / max(k0, 1e-12))
        return float(np.mean(vals))

    res = {"checkpoints": list(checkpoints), "d_model": d, "blocks": [], "n_train": half, "n_test": n_pairs - half,
           "kl_bt_bb": float(r7._kl(BT, BB).mean()), "kl_bt_original": float(np.mean([
               kl_to_bt(te, baseB[te, m - 1], c).mean().item() for c, m in enumerate(checkpoints)]))}
    for li in range(n_blocks):
        X = col["resid"][li + 1]
        Qdec = _orth(ridge_fit(X, Y, alpha=1.0)["W"])                                    # decoder directions (d, 3)
        Xc = X - X.mean(0)
        E = np.linalg.lstsq(Ys, Xc, rcond=None)[0]                                       # encoder directions (3, d)
        Qenc = _orth(E.T)
        T = lambda M: torch.as_tensor(M, dtype=torch.float32, device=device)  # noqa: E731
        for mode in modes:
            row = {"block": li, "mode": mode}
            row["decoder3"] = transfer(te, T(Qdec), li, mode)
            row["encoder3"] = transfer(te, T(Qenc), li, mode)
            row["full"] = transfer(te, torch.eye(d, device=device), li, mode)
            for k in ks:
                row[f"random{k}"] = float(np.mean([transfer(te, T(_orth(rng.standard_normal((d, k)))), li, mode)
                                                   for _ in range(3)]))
                M = torch.nn.Parameter(torch.as_tensor(rng.standard_normal((d, k)) * 0.1, dtype=torch.float32, device=device))
                opt = torch.optim.Adam([M], lr=lr)
                with torch.enable_grad():
                    for _ in range(steps):
                        Q, _ = torch.linalg.qr(M)
                        loss = sum(kl_to_bt(tr, patched_lp(tr, li, Q, m, mode), c).mean() for c, m in enumerate(checkpoints))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    Q, _ = torch.linalg.qr(M)
                row[f"das{k}"] = transfer(te, Q, li, mode)
                row[f"das{k}_train"] = transfer(tr, Q, li, mode)
                Qn = Q.cpu().numpy()
                row[f"das{k}_overlap_decoder"] = float(np.linalg.norm(Qn.T @ Qdec) ** 2 / Qdec.shape[1])
                row[f"das{k}_overlap_encoder"] = float(np.linalg.norm(Qn.T @ Qenc) ** 2 / Qenc.shape[1])
                row[f"das{k}_overlap_random_expected"] = float(k / d)
            res["blocks"].append(row)
    for p in model.parameters():
        p.requires_grad_(True)
    return res


# ------------------------------------------------------------------------------------------- M1 confidence intervals
def misspec_ci(model, model_dcfg: dict, datasets: dict, bayes: dict, edges, checkpoints, device, n_boot: int = 2000,
               seed: int = 0) -> dict:
    own = kernel_of(model_dcfg)
    idx = np.array(checkpoints) - 1
    rng = np.random.default_rng(seed)
    out = {"own_family": own, "data": {}}
    for fam, b in datasets.items():
        q = np.exp(r7.model_outputs(model, b, device)["logp"][:, idx])
        P_own, P_true = bayes[(own, fam)]["P"], bayes[(fam, fam)]["P"]
        ko = r7._kl(P_own, q).mean(1)                    # per sequence, averaged over context lengths
        kt = r7._kl(P_true, q).mean(1)
        S = len(ko)
        ii = rng.integers(S, size=(n_boot, S))
        diff = (kt - ko)[ii].mean(1)
        ratio = ko[ii].mean(1) / np.maximum(kt[ii].mean(1), 1e-12)
        out["data"][fam] = {
            "misspecified": fam != own, "n_seqs": int(S),
            "kl_own": float(ko.mean()), "kl_true": float(kt.mean()),
            "diff_true_minus_own": float((kt - ko).mean()), "diff_ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
            "ratio_own_over_true": float(ko.mean() / max(kt.mean(), 1e-12)),
            "ratio_ci95": [float(x) for x in np.percentile(ratio, [2.5, 97.5])],
            "share_seqs_closer_to_own": float((ko < kt).mean()),
            "ess_median_own_prior": [float(np.median(bayes[(own, fam)]["ess"][:, c])) for c in range(len(idx))],
            "ess_median_true_prior": [float(np.median(bayes[(fam, fam)]["ess"][:, c])) for c in range(len(idx))],
        }
    return out
