"""Round 7 for the Hawkes paper: three tests that turn "the posterior is decodable" into "the model is a Bayesian".

M  Misspecification. Run each model on sequences from the OTHER kernel family. A network that implements Bayesian
   inference under its training prior should forecast like the Bayes posterior predictive of its own model class
   (KL to "Bayes-own" small), not like the Bayes forecaster of the true class ("Bayes-true"). Its decoded branching
   ratio should track the own-class posterior mean of alpha, not the true alpha.
I  Posterior interchange. Swap the posterior subspace (span of the probe directions for the posterior means) of
   sequence B with that of sequence A. The prediction of a "posterior + excitation" computation: the forecast becomes
   the Bayes forecast that uses A's posterior weights on B's history, sum_k w_k^A P(next | H_B, theta_k).
K  Kernel bank. Is the history summarised by exponentially decayed event counts at several fixed rates,
   S_b(t_i) = sum_{j<=i} exp(-b (t_i - t_j))? (1) do these statistics suffice for the posterior means, (2) are they
   linearly decodable from the residual stream, (3) does replacing a block's output by its linear reconstruction from
   them preserve the forecast (compared with count-and-time features only)?
"""
from __future__ import annotations

import numpy as np
import torch

from ..core.probes import mlp_probe, r2_per_output, ridge_cv, ridge_fit
from .baselines import _loglik_checkpoints, bin_probs_powerlaw_torch
from .posterior import loglik_powerlaw, posterior_stats, prior_draws
from .simulate import oracle_bin_probs
from .task import collect, kernel_of, make_batch, oracle_probs

PARAMS = ("log_mu", "alpha", "log_beta")


def _kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, None)
    Q = np.clip(Q, 1e-12, None)
    return (P * (np.log(P) - np.log(Q))).sum(-1)


# ------------------------------------------------------------------------------------------- shared Bayes engine
def bayes_dists(family_dcfg: dict, times: np.ndarray, checkpoints, edges: np.ndarray, K: int, seed, device,
                weight_floor: float = 1e-10, return_draws: bool = False) -> dict:
    """Posterior predictive over the next interval after m events, under the prior of ``family_dcfg``.

    times (S, N). Returns P (S, C, bins), posterior means {param: (S, C)}, ess (S, C); with return_draws, the
    per-sequence log-likelihoods (K, C) and the prior draws (for the interchange test)."""
    prior = prior_draws(family_dcfg, K, seed)
    S, C = times.shape[0], len(checkpoints)
    nb = len(edges) - 1
    P = np.zeros((S, C, nb))
    post = {k: np.zeros((S, C)) for k in PARAMS}
    ess = np.zeros((S, C))
    theta = {"log_mu": np.log(prior["mu"]), "alpha": prior["alpha"], "log_beta": np.log(prior["beta"])}
    lls = np.zeros((S, K, C)) if return_draws else None
    pl = kernel_of(family_dcfg) == "powerlaw"
    dev = str(device) if not isinstance(device, str) else device
    for s in range(S):
        t = times[s]
        if pl:
            ll_all = loglik_powerlaw(t, prior, max(checkpoints), dev)
        else:
            lld, Ed = _loglik_checkpoints(t, prior["mu"], prior["alpha"], prior["beta"], checkpoints)
        for c, m in enumerate(checkpoints):
            ll = ll_all[:, m - 1] if pl else lld[m]
            if return_draws:
                lls[s, :, c] = ll
            w = np.exp(ll - ll.max())
            w /= w.sum()
            ess[s, c] = 1.0 / np.sum(w ** 2)
            for k in PARAMS:
                post[k][s, c] = theta[k] @ w
            keep = w > weight_floor * w.max()
            if pl:
                sub = {k: np.asarray(v)[keep] for k, v in prior.items()}
                Pk = bin_probs_powerlaw_torch(t[:m], sub, edges, dev)
            else:
                Pk = oracle_bin_probs(prior["mu"][keep], Ed[m][keep], prior["beta"][keep], edges)
            P[s, c] = (w[keep] / w[keep].sum()) @ Pk
    out = {"P": P, "post": post, "ess": ess}
    if return_draws:
        out["lls"], out["prior"] = lls, prior
    return out


@torch.no_grad()
def model_outputs(model, b: dict, device, layers=None) -> dict:
    model.eval()
    out = model(feats=torch.from_numpy(b["feats"]).to(device), pos=torch.from_numpy(b["times"]).to(device),
                return_resid=layers is not None)
    res = {"logp": torch.log_softmax(out["logits"].float(), -1).cpu().numpy()}
    if layers is not None:
        res["resid"] = {li: out["resid"][li].float().cpu().numpy() for li in layers}
    return res


# ------------------------------------------------------------------------------------------- M: misspecification
def misspecification(model, model_dcfg: dict, datasets: dict, bayes: dict, edges, checkpoints, device,
                     probe_layer: int, fit: dict) -> dict:
    """datasets: {family: batch with truth}; bayes: {(prior_family, data_family): bayes_dists output}.
    fit: in-distribution {"resid": (n, d), "alpha_post": (n,)} to train the alpha probe at ``probe_layer``."""
    own = kernel_of(model_dcfg)
    W = ridge_fit(fit["resid"], fit["alpha_post"], alpha=1.0)
    res = {"own_family": own, "probe_layer": probe_layer, "data": {}}
    idx = np.array(checkpoints) - 1
    for fam, b in datasets.items():
        mo = model_outputs(model, b, device, layers=[probe_layer])
        q = np.exp(mo["logp"][:, idx])                                       # (S, C, bins)
        P_own, P_true = bayes[(own, fam)]["P"], bayes[(fam, fam)]["P"]
        P_orc = oracle_probs({"kernel": fam}, b, edges)[:, idx]
        y = b["labels"][:, idx]
        ce = lambda Pm: float(-np.log(np.clip(np.take_along_axis(Pm, y[..., None], -1)[..., 0], 1e-300, None)).mean())  # noqa: E731
        rows = []
        for c, m in enumerate(checkpoints):
            rows.append({"context": int(m), "kl_own": float(_kl(P_own[:, c], q[:, c]).mean()),
                         "kl_true": float(_kl(P_true[:, c], q[:, c]).mean()),
                         "kl_own_true": float(_kl(P_true[:, c], P_own[:, c]).mean())})
        dec = mo["resid"][probe_layer][:, idx].reshape(-1, W["W"].shape[0]) @ W["W"][:, 0] + W["b"][0]
        a_own = bayes[(own, fam)]["post"]["alpha"].reshape(-1)
        a_true = np.repeat(b["alpha"], len(idx))
        r2 = lambda y_, yh: float(r2_per_output(y_[:, None], yh[:, None])[0])  # noqa: E731
        res["data"][fam] = {
            "misspecified": fam != own,
            "rows": rows,
            "kl_own": float(np.mean([r["kl_own"] for r in rows])),
            "kl_true": float(np.mean([r["kl_true"] for r in rows])),
            "ce_model": ce(q), "ce_bayes_own": ce(P_own), "ce_bayes_true": ce(P_true),
            "ce_oracle": ce(P_orc),
            "alpha_r2_decoded_vs_own_posterior": r2(a_own, dec),
            "alpha_r2_decoded_vs_true": r2(a_true, dec),
            "alpha_r2_own_posterior_vs_true": r2(a_true, a_own),
            "alpha_corr_decoded_own": float(np.corrcoef(dec, a_own)[0, 1]),
            "alpha_corr_decoded_true": float(np.corrcoef(dec, a_true)[0, 1]),
        }
    return res


def fit_alpha_probe(model, dcfg: dict, edges, n_seqs: int, seed, device, layer: int, K: int, positions) -> dict:
    """In-distribution residuals at the checkpoints and the posterior mean of alpha there."""
    b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 71], edges, with_truth=True)
    idx = np.array(positions) - 1
    mo = model_outputs(model, b, device, layers=[layer])
    post = posterior_stats(dcfg, b["times64"], idx, K=K, seed=seed + 1,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    return {"resid": mo["resid"][layer][:, idx].reshape(-1, mo["resid"][layer].shape[-1]),
            "alpha_post": post["alpha"]["mean"].reshape(-1)}


# ------------------------------------------------------------------------------------------- I: interchange
def _orth(W: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(W)
    return q


@torch.no_grad()
def interchange(model, dcfg: dict, edges, n_fit: int, n_pairs: int, seed, device, layers=(1, 2, 3),
                checkpoints=(64, 128), K: int = 20000, query_from: int = 48, with_sd: bool = False) -> dict:
    """Exponential kernel. See module docstring (test I)."""
    assert kernel_of(dcfg) == "exp"
    model.eval()
    col = collect(model, dcfg, edges, n_fit, seed, device, query_from=query_from)
    post = posterior_stats(dcfg, col["times"], col["query_idx"], K=K, seed=seed + 1,
                           device="cuda" if torch.cuda.is_available() else "cpu")
    Y = np.stack([post[k]["mean"].reshape(-1) for k in PARAMS], 1)
    if with_sd:
        Y = np.concatenate([Y, np.stack([post[k]["sd"].reshape(-1) for k in PARAMS], 1)], 1)
    rng = np.random.default_rng(seed + 2)
    A = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 501], edges, with_truth=True)
    B = make_batch({**dcfg, "batch_size": n_pairs}, [seed, 502], edges, with_truth=True)
    bayesA = bayes_dists(dcfg, A["times64"], checkpoints, edges, K, seed + 3, device, return_draws=True)
    prior = bayesA["prior"]
    res = {"layers": {}, "checkpoints": list(checkpoints), "dim": int(Y.shape[1])}
    resA = model_outputs(model, A, device, layers=[li + 1 for li in layers])["resid"]
    base = model_outputs(model, B, device)["logp"]
    # Bayes forecasts on B: own weights (BB) and A's weights (BT), from the same prior draws
    BT = np.zeros((n_pairs, len(checkpoints), len(edges) - 1))
    BB = np.zeros_like(BT)
    for s in range(n_pairs):
        lld, Ed = _loglik_checkpoints(B["times64"][s], prior["mu"], prior["alpha"], prior["beta"], checkpoints)
        for c, m in enumerate(checkpoints):
            for target, ll in ((BT, bayesA["lls"][s, :, c]), (BB, lld[m])):
                w = np.exp(ll - ll.max())
                w /= w.sum()
                keep = w > 1e-10 * w.max()
                Pk = oracle_bin_probs(prior["mu"][keep], Ed[m][keep], prior["beta"][keep], edges)
                target[s, c] = (w[keep] / w[keep].sum()) @ Pk
    for li in layers:
        Wf = ridge_fit(col["resid"][li + 1], Y, alpha=1.0)["W"]          # (d, k) decoding directions
        Q = _orth(Wf)
        Qr = _orth(rng.standard_normal(Wf.shape))
        row = {}
        for cname, Qm in (("posterior", Q), ("random", Qr)):
            P_ = torch.as_tensor(Qm @ Qm.T, dtype=torch.float32, device=device)
            for c, m in enumerate(checkpoints):
                xa = torch.as_tensor(resA[li + 1][:, :m], device=device)

                def hook(i, x, li=li, m=m, xa=xa, P_=P_):
                    if i == li:
                        x = x.clone()
                        x[:, :m] = x[:, :m] + (xa.to(x.dtype) - x[:, :m]) @ P_.to(x.dtype)
                    return x

                out = model(feats=torch.from_numpy(B["feats"]).to(device), pos=torch.from_numpy(B["times"]).to(device),
                            edit=hook)
                q = torch.softmax(out["logits"][:, m - 1].float(), -1).cpu().numpy()
                q0 = np.exp(base[:, m - 1])
                kl_bt_q, kl_bt_q0 = _kl(BT[:, c], q).mean(), _kl(BT[:, c], q0).mean()
                row[f"{cname}_m{m}"] = {"kl_transplant_patched": float(kl_bt_q), "kl_transplant_original": float(kl_bt_q0),
                                        "kl_own_patched": float(_kl(BB[:, c], q).mean()),
                                        "kl_own_original": float(_kl(BB[:, c], q0).mean()),
                                        "kl_transplant_own": float(_kl(BT[:, c], BB[:, c]).mean()),
                                        "transfer": float(1 - kl_bt_q / max(kl_bt_q0, 1e-12))}
        row["transfer_posterior"] = float(np.mean([row[f"posterior_m{m}"]["transfer"] for m in checkpoints]))
        row["transfer_random"] = float(np.mean([row[f"random_m{m}"]["transfer"] for m in checkpoints]))
        res["layers"][int(li)] = row
    best = max(res["layers"], key=lambda k: res["layers"][k]["transfer_posterior"])
    res["best_layer"] = int(best)
    return res


# ------------------------------------------------------------------------------------------- K: kernel bank
DEFAULT_RATES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def bank_features(times: np.ndarray, feats: np.ndarray, rates=DEFAULT_RATES) -> dict:
    """times (S, N), feats (S, N, 2) -> {"bank": (S, N, 2R), "basic": (S, N, 4)} (per position, causal)."""
    S, N = times.shape
    R = len(rates)
    Sb = np.zeros((S, N, R))
    r = np.asarray(rates)[None, :]
    acc = np.zeros((S, R))
    prev = np.zeros(S)
    for i in range(N):
        dt = (times[:, i] - prev)[:, None]
        acc = 1.0 + acc * np.exp(-r * dt) if i > 0 else np.ones((S, R))
        Sb[:, i] = acc
        prev = times[:, i]
    idx = np.arange(N)[None, :].repeat(S, 0)
    basic = np.concatenate([feats, np.log1p(idx)[..., None], np.log1p(times)[..., None]], -1)
    bank = np.concatenate([np.log1p(Sb), Sb / (1 + idx[..., None])], -1)
    return {"bank": bank, "basic": basic, "rates": list(rates)}


def kernel_bank(model, dcfg: dict, edges, n_seqs: int, seed, device, K: int = 20000, query_from: int = 48,
                rates=DEFAULT_RATES, n_test: int = 400, min_damage: float = 0.01) -> dict:
    model.eval()
    col = collect(model, dcfg, edges, n_seqs, seed, device, query_from=query_from)
    q = col["query_idx"]
    b_feats = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 77], edges, with_truth=True)   # same batch as collect
    fb = bank_features(b_feats["times64"], b_feats["feats"], rates)
    Xb = np.concatenate([fb["bank"], fb["basic"]], -1)[:, q].reshape(-1, len(rates) * 2 + 4)
    Xc = fb["basic"][:, q].reshape(-1, 4)
    post = posterior_stats(dcfg, col["times"], q, K=K, seed=seed + 1, device="cuda" if torch.cuda.is_available() else "cpu")
    res = {"rates": list(rates), "sufficiency": {}, "representation": [], "reconstruction": []}
    groups = col["groups"]
    mdev = "cuda" if torch.cuda.is_available() else "cpu"
    for k in PARAMS:
        y = post[k]["mean"].reshape(-1)
        best_resid = max(float(ridge_cv(X, y, groups=groups, alphas=(0.1, 1, 10, 100))["r2"][0]) for X in col["resid"])
        res["sufficiency"][k] = {
            "bank_linear": float(ridge_cv(Xb, y, groups=groups, alphas=(0.1, 1, 10, 100))["r2"][0]),
            "bank_mlp": float(mlp_probe(Xb, y, groups=groups, epochs=150, device=mdev)["r2"][0]),
            "count_time_mlp": float(mlp_probe(Xc, y, groups=groups, epochs=150, device=mdev)["r2"][0]),
            "residual_best_linear": best_resid}
    for li, X in enumerate(col["resid"]):
        r2 = ridge_cv(X, fb["bank"][:, q, :len(rates)].reshape(-1, len(rates)), groups=groups, alphas=(0.1, 1, 10, 100))["r2"]
        res["representation"].append({"layer": li, **{f"rate_{b}": float(v) for b, v in zip(rates, r2)}})
    # reconstruction: replace the residual after block l (all positions) by a linear map of features
    test = make_batch({**dcfg, "batch_size": n_test}, [seed, 78], edges, with_truth=True)
    ft = bank_features(test["times64"], test["feats"], rates)
    fit_b = make_batch({**dcfg, "batch_size": n_seqs}, [seed, 79], edges, with_truth=True)
    ff = bank_features(fit_b["times64"], fit_b["feats"], rates)
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
    perm = np.random.default_rng(seed + 5).permutation(n_test)
    for li in range(len(full_fit) - 2):        # after blocks 0 .. L-2 (the last block's output is the readout)
        Hf = full_fit[li + 1].float().cpu().numpy().reshape(-1, full_fit[li + 1].shape[-1])
        row = {"block": li, "ce_original": base}
        for name, Ff, Ft in (("bank", np.concatenate([ff["bank"], ff["basic"]], -1), np.concatenate([ft["bank"], ft["basic"]], -1)),
                             ("count_time", ff["basic"], ft["basic"]),
                             ("bank_shuffled", np.concatenate([ff["bank"], ff["basic"]], -1),
                              np.concatenate([ft["bank"][perm], ft["basic"]], -1))):
            Wr = ridge_fit(Ff.reshape(-1, Ff.shape[-1]), Hf, alpha=1.0)
            Hhat = torch.as_tensor(Ft.reshape(-1, Ft.shape[-1]) @ Wr["W"] + Wr["b"], dtype=torch.float32,
                                   device=device).reshape(n_test, Ft.shape[1], -1)

            def hook(i, x, li=li, Hhat=Hhat):
                return Hhat.to(x.dtype) if i == li else x

            row[f"ce_{name}"] = ce(fwd(hook))
        damage = row["ce_count_time"] - base
        row["damage_count_time"] = float(damage)
        # undefined when count+time features alone already reproduce the block (nothing left to recover)
        row["recovered"] = float((row["ce_count_time"] - row["ce_bank"]) / damage) if damage >= min_damage else None
        res["reconstruction"].append(row)
    return res
