"""Round 9 for the Hawkes paper: two checks a reviewer will ask for.

A  Pooling heads are necessary. Round 8 found heads with almost exactly uniform attention over past events (total
   variation < 0.1), and pooled log-traces decodable right after them. Mean-ablate those heads (replace each head's
   output by its average) and compare with ablating the same number of other heads: late-context cross-entropy, and how
   much pooled-trace information their block's output still holds (probes refit on the ablated model).
D  The DAS subspace is not an illusion. (1) Wrong target: train a 3-dim DAS subspace to make B's forecast the Bayes
   forecast with the posterior of an unrelated sequence C while patching from A; if subspace patching could produce
   anything, this would also succeed. (2) The 3 DAS coordinates themselves predict the posterior means. (3) The
   subspaces found at different blocks are aligned. The fitted subspaces are saved.
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.probes import ridge_cv, ridge_fit
from . import r7, r8
from .posterior import posterior_stats
from .task import collect, make_batch

PARAMS = r7.PARAMS


# ------------------------------------------------------------------------------------------- A: head ablation
def _attn_modules(model):
    return [blk.attn for blk in model.blocks]


@torch.no_grad()
def head_means(model, b: dict, device) -> dict:
    """Average output of every head over all positions and sequences of batch b: {(block, head): (hd,)}."""
    mods = _attn_modules(model)
    for m in mods:
        m.capture_heads = True
    model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device))
    out = {}
    for bi, m in enumerate(mods):
        y = m.head_out.float()                                 # (B, H, T, hd)
        for h in range(y.shape[1]):
            out[(bi, h)] = y[:, h].mean((0, 1))
        m.capture_heads = False
        m.head_out = None
    return out


def set_ablation(model, heads, means: dict | None):
    mods = _attn_modules(model)
    for m in mods:
        m.head_replace = {}
    for bi, h in heads:
        mods[bi].head_replace[h] = None if means is None else means[(bi, h)]


def pooling_heads(attention_rows: list, tv: float = 0.25) -> list[tuple]:
    return [(a["block"], a["head"]) for a in attention_rows if a["tv_uniform"] < tv]


def head_ablation(model, dcfg: dict, edges, heads: list, n_seqs: int, seed, device, K: int = 20000, n_random: int = 10,
                  query_from: int = 48) -> dict:
    """Mean-ablate ``heads`` and n_random random head sets of the same size, in the same blocks where possible and never
    containing a pooling head. Metrics: late-context CE; pooled-trace R2 (refit) after the last ablated head's block; posterior R2
    at the final layer."""
    model.eval()
    ref = make_batch({**dcfg, "batch_size": 256}, [seed, 90], edges, with_truth=True)
    means = head_means(model, ref, device)
    test = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 91], edges, with_truth=True)
    q = np.arange(query_from, test["feats"].shape[1] - 1)
    tf = r8.trace_features(test["times64"], test["feats"])
    Yp = tf["pooled"][:, q].reshape(-1, tf["pooled"].shape[-1])
    post = posterior_stats(dcfg, test["times64"], q, K=K, seed=seed + 1, device="cuda" if torch.cuda.is_available() else "cpu")
    Ypost = np.stack([post[k]["mean"].reshape(-1) for k in PARAMS], 1)
    groups = np.repeat(np.arange(n_seqs), len(q))
    y = test["labels"]
    late = np.zeros_like(y, dtype=bool)
    late[:, 32:-1] = True
    n_blocks, n_heads = len(model.blocks), model.blocks[0].attn.h

    @torch.no_grad()
    def evaluate(hs):
        set_ablation(model, hs, means)
        out = model(feats=torch.from_numpy(test["feats"]).to(device), pos=torch.from_numpy(test["times"]).to(device),
                    return_resid=True)
        set_ablation(model, [], None)
        lp = torch.log_softmax(out["logits"].float(), -1).cpu().numpy()
        ce = float(-np.take_along_axis(lp, np.maximum(y, 0)[..., None], -1)[..., 0][late].mean())
        layer = min(max([b for b, _ in hs], default=0) + 1, n_blocks)       # residual right after their block
        R = out["resid"][layer].float().cpu().numpy()[:, q].reshape(-1, out["resid"][layer].shape[-1])
        rp = float(np.nanmean(ridge_cv(R, Yp, groups=groups, alphas=(0.1, 1, 10, 100))["r2"]))
        rpost = [float(x) for x in ridge_cv(out["resid"][-1].float().cpu().numpy()[:, q].reshape(-1, R.shape[-1]), Ypost,
                                            groups=groups, alphas=(0.1, 1, 10, 100))["r2"]]
        return {"ce": ce, "pooled_r2": rp, "posterior_r2_final": rpost, "probe_layer": int(layer)}

    base = evaluate([])
    res = {"heads": [list(h) for h in heads], "baseline": base, "pooling": evaluate(heads) if heads else None, "random": []}
    rng = np.random.default_rng(seed + 7)
    pool = set(map(tuple, heads))
    for _ in range(n_random if heads else 0):
        pick = []
        for bi, _h in heads:                                   # same block for each, never a pooling head
            cands = [(bi, h) for h in range(n_heads) if (bi, h) not in pool and (bi, h) not in pick]
            if not cands:
                cands = [(b2, h) for b2 in range(n_blocks) for h in range(n_heads) if (b2, h) not in pool and (b2, h) not in pick]
            if cands:
                pick.append(cands[int(rng.integers(len(cands)))])
        if len(pick) == len(heads):
            res["random"].append({"heads": [list(h) for h in pick], **evaluate(pick)})
    if heads and not res["random"]:
        res["random"].append({"heads": [], **base})                 # nothing else to ablate (tiny models only)
    if heads:
        dce = res["pooling"]["ce"] - base["ce"]
        rce = [r["ce"] - base["ce"] for r in res["random"]]
        res["summary"] = {"dce_pooling": float(dce), "dce_random_mean": float(np.mean(rce)), "dce_random_max": float(np.max(rce)),
                          "pooled_r2_drop_pooling": float(base["pooled_r2"] - res["pooling"]["pooled_r2"]),
                          "pooled_r2_drop_random_mean": float(np.mean([base["pooled_r2"] - r["pooled_r2"] for r in res["random"]]))}
    return res


# ------------------------------------------------------------------------------------------- D: DAS controls
def _orth(W):
    return np.linalg.qr(W)[0]


def das_controls(model, dcfg: dict, edges, n_fit: int, n_pairs: int, seed, device, blocks=(1, 2), k: int = 3,
                 checkpoints=(64, 128), K: int = 20000, steps: int = 300, lr: float = 0.02, query_from: int = 48) -> dict:
    """Prefix-mode DAS at the given blocks: right target (A's posterior on B) and wrong target (C's posterior on B,
    patching from A); posterior R2 from the k DAS coordinates; alignment of the subspaces across blocks."""
    model.eval()
    A = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 601], edges, with_truth=True)
    B = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 602], edges, with_truth=True)
    C = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 603], edges, with_truth=True)
    BT, _ = r8._bayes_transplant(dcfg, A, B, checkpoints, edges, K, seed + 3, device)
    BC, _ = r8._bayes_transplant(dcfg, C, B, checkpoints, edges, K, seed + 5, device)
    with torch.no_grad():
        residA = [r.float() for r in model(feats=torch.from_numpy(A["feats"]).to(device),
                                           pos=torch.from_numpy(A["times"]).to(device), return_resid=True)["resid"]]
        baseB = torch.log_softmax(model(feats=torch.from_numpy(B["feats"]).to(device),
                                        pos=torch.from_numpy(B["times"]).to(device))["logits"].float(), -1)
    fB, tB = torch.from_numpy(B["feats"]).to(device), torch.from_numpy(B["times"]).to(device)
    half = n_pairs // 2
    tr, te = slice(0, half), slice(half, n_pairs)
    targets = {"right": torch.as_tensor(BT, dtype=torch.float32, device=device).clamp_min(1e-12),
               "wrong": torch.as_tensor(BC, dtype=torch.float32, device=device).clamp_min(1e-12)}
    d = residA[0].shape[-1]
    rng = np.random.default_rng(seed + 9)
    for p in model.parameters():
        p.requires_grad_(False)

    def patched_lp(sl, li, Q, m):
        xa = residA[li + 1][sl]

        def hook(i, x):
            if i == li:
                x = x.clone()
                diff = xa[:, :m].to(x.dtype) - x[:, :m]
                x[:, :m] = x[:, :m] + (diff @ Q.to(x.dtype)) @ Q.to(x.dtype).T
            return x

        return torch.log_softmax(model(feats=fB[sl], pos=tB[sl], edit=hook)["logits"][:, m - 1].float(), -1)

    def kl(P, lp):
        return (P * (P.log() - lp)).sum(-1)

    def transfer(sl, Q, li, tgt):
        P = targets[tgt]
        vals = []
        with torch.no_grad():
            for c, m in enumerate(checkpoints):
                kp = kl(P[sl, c], patched_lp(sl, li, Q, m)).mean().item()
                k0 = kl(P[sl, c], baseB[sl, m - 1]).mean().item()
                vals.append(1 - kp / max(k0, 1e-12))
        return float(np.mean(vals))

    def fit(li, tgt):
        M = torch.nn.Parameter(torch.as_tensor(rng.standard_normal((d, k)) * 0.1, dtype=torch.float32, device=device))
        opt = torch.optim.Adam([M], lr=lr)
        P = targets[tgt]
        with torch.enable_grad():
            for _ in range(steps):
                Q, _ = torch.linalg.qr(M)
                loss = sum(kl(P[tr, c], patched_lp(tr, li, Q, m)).mean() for c, m in enumerate(checkpoints))
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            return torch.linalg.qr(M)[0]

    col = collect(model, dcfg, edges, n_fit, seed, device, query_from=query_from)
    post = posterior_stats(dcfg, col["times"], col["query_idx"], K=K, seed=seed + 1,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    Y = np.stack([post[p]["mean"].reshape(-1) for p in PARAMS], 1)
    res = {"k": k, "blocks": [], "subspaces": {}}
    Qs = {}
    for li in blocks:
        Qr = fit(li, "right")
        Qw = fit(li, "wrong")
        Qs[li] = Qr.cpu().numpy()
        X = col["resid"][li + 1]
        Xc = X - X.mean(0)
        Qrand = _orth(rng.standard_normal((d, k)))
        r2 = lambda Z: [float(x) for x in ridge_cv(Z, Y, groups=col["groups"], alphas=(0.01, 0.1, 1, 10))["r2"]]  # noqa: E731
        row = {"block": li,
               "right_target_transfer": transfer(te, Qr, li, "right"),
               "wrong_target_transfer": transfer(te, Qw, li, "wrong"),
               "right_subspace_on_wrong_target": transfer(te, Qr, li, "wrong"),
               "random_subspace_right_target": transfer(te, torch.as_tensor(Qrand, dtype=torch.float32, device=device), li, "right"),
               "posterior_r2_from_das_coords": r2(Xc @ Qs[li]),
               "posterior_r2_from_random_coords": r2(Xc @ Qrand),
               "posterior_r2_full_residual": r2(X)}
        res["blocks"].append(row)
        res["subspaces"][str(li)] = Qs[li].tolist()
    ks = sorted(Qs)
    res["alignment"] = [{"blocks": [a, b], "overlap": float(np.linalg.norm(Qs[a].T @ Qs[b]) ** 2 / k)}
                        for i, a in enumerate(ks) for b in ks[i + 1:]]
    res["alignment_random_expected"] = float(k / d)
    for p in model.parameters():
        p.requires_grad_(True)
    return res
