"""Tests for the round-7 to round-9 Hawkes analyses (misspecification, decayed-count features, pooled traces,
interchange / DAS, head ablation) and the optional FIM-PP comparison."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mechlab.hawkes import r7, r8, r9
from mechlab.hawkes import task as ht
from mechlab.hawkes.simulate import bin_edges


def _hawkes_model(dcfg):
    torch.manual_seed(0)
    return ht.build_model({"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}, 48).eval()


def test_bank_features_match_direct_sum():
    rng = np.random.default_rng(0)
    times = np.cumsum(rng.exponential(1.0, size=(3, 30)), 1)
    feats = rng.standard_normal((3, 30, 2)).astype(np.float32)
    fb = r7.bank_features(times, feats, rates=(0.5, 2.0))
    i, s = 17, 1
    direct = np.exp(-0.5 * (times[s, i] - times[s, :i + 1])).sum()
    assert np.isclose(np.expm1(fb["bank"][s, i, 0]), direct)


def test_bayes_dists_are_distributions_and_match_baseline():
    edges = bin_edges(48)
    dcfg = {"n_events": 40, "batch_size": 4, "params": {}}
    b = ht.make_batch(dcfg, [0, 1], edges, with_truth=True)
    res = r7.bayes_dists(dcfg, b["times64"], [4, 16], edges, 2000, 3, "cpu")
    assert np.allclose(res["P"].sum(-1), 1, atol=1e-6)
    assert ((res["post"]["alpha"] >= 0) & (res["post"]["alpha"] <= 0.9)).all()
    pl = {"n_events": 40, "kernel": "powerlaw", "params": {"mu": [0.2, 2.0], "alpha": [0.0, 0.9], "c": [0.1, 2.0], "gamma": [1.5, 3.0]}}
    res2 = r7.bayes_dists(pl, b["times64"], [4, 16], edges, 500, 3, "cpu")
    assert np.allclose(res2["P"].sum(-1), 1, atol=1e-6)


def test_r7_analyses_run_on_tiny_model():
    edges = bin_edges(48)
    dcfg = {"n_events": 64, "batch_size": 4, "params": {}}
    model = _hawkes_model(dcfg)
    ic = r7.interchange(model, dcfg, edges, 6, 5, 0, "cpu", layers=(1,), checkpoints=(16, 32), K=300)
    assert "transfer_posterior" in ic["layers"][1]
    kb = r7.kernel_bank(model, dcfg, edges, 10, 0, "cpu", K=300, n_test=4)
    assert set(kb["sufficiency"]) == set(r7.PARAMS) and kb["reconstruction"]
    pl = {"n_events": 64, "kernel": "powerlaw", "params": {"mu": [0.2, 2.0], "alpha": [0.0, 0.9], "c": [0.1, 2.0], "gamma": [1.5, 3.0]}}
    fams = {"exp": dcfg, "powerlaw": pl}
    data = {f: ht.make_batch({**d, "batch_size": 4}, [9, i], edges, with_truth=True) for i, (f, d) in enumerate(fams.items())}
    bayes = {(pf, df): r7.bayes_dists(pd, data[df]["times64"], [8, 32], edges, 300, 1, "cpu") for pf, pd in fams.items() for df in fams}
    fit = r7.fit_alpha_probe(model, dcfg, edges, 6, 0, "cpu", 2, 300, [8, 32])
    ms = r7.misspecification(model, dcfg, data, bayes, edges, [8, 32], "cpu", 2, fit)
    assert ms["data"]["powerlaw"]["misspecified"] and not ms["data"]["exp"]["misspecified"]


def test_pooled_traces_match_direct_sum():
    rng = np.random.default_rng(0)
    times = np.cumsum(rng.exponential(1.0, size=(2, 25)), 1)
    feats = rng.standard_normal((2, 25, 2)).astype(np.float32)
    tf = r8.trace_features(times, feats, rates=(0.5, 2.0), cs=(0.1, 1.0))
    s, i, bi, ci = 1, 17, 1, 0                           # rate 2.0, offset 0.1
    R = [np.exp(-2.0 * (times[s, j] - times[s, :j])).sum() for j in range(i + 1)]
    want = np.mean(np.log(0.1 + np.array(R)))
    assert np.isclose(tf["pooled"][s, i, ci * 2 + bi], want, rtol=1e-4)


def test_r8_analyses_run_on_tiny_model():
    torch.manual_seed(0)
    edges = bin_edges(48)
    dcfg = {"n_events": 64, "batch_size": 4, "params": {}}
    model = ht.build_model({"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}, 48).eval()
    pt = r8.pooled_tests(model, dcfg, edges, 10, 0, "cpu", K=300, n_test=6, query_from=24, mlp_epochs=3)
    assert set(pt["sufficiency"]) == set(r8.PARAMS) and len(pt["representation"]) == 4 and pt["attention"]
    das = r8.interchange_das(model, dcfg, edges, 6, 8, 0, "cpu", checkpoints=(16, 32), K=300, ks=(3,), steps=2)
    assert len(das["blocks"]) == 3 * 2 and all("das3" in b for b in das["blocks"])
    assert all(p.requires_grad for p in model.parameters())
    pl = {"n_events": 64, "kernel": "powerlaw", "params": {"mu": [0.2, 2.0], "alpha": [0.0, 0.9], "c": [0.1, 2.0], "gamma": [1.5, 3.0]}}
    fams = {"exp": dcfg, "powerlaw": pl}
    data = {f: ht.make_batch({**d, "batch_size": 5}, [9, i], edges, with_truth=True) for i, (f, d) in enumerate(fams.items())}
    bayes = {(pf, df): r7.bayes_dists(pd, data[df]["times64"], [8, 32], edges, 300, 1, "cpu") for pf, pd in fams.items() for df in fams}
    ci = r8.misspec_ci(model, dcfg, data, bayes, edges, [8, 32], "cpu", n_boot=20)
    d = ci["data"]["powerlaw"]
    assert d["misspecified"] and d["diff_ci95"][0] <= d["diff_true_minus_own"] <= d["diff_ci95"][1]


def test_head_hooks():
    torch.manual_seed(0)
    edges = bin_edges(48)
    dcfg = {"n_events": 40, "batch_size": 6, "params": {}}
    model = ht.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, 48).eval()
    b = ht.make_batch(dcfg, [0, 1], edges)
    x = dict(feats=torch.from_numpy(b["feats"]), pos=torch.from_numpy(b["times"]))
    with torch.no_grad():
        base = model(**x)["logits"]
        means = r9.head_means(model, b, "cpu")
        assert all(m.capture_heads is False and not m.head_replace for m in r9._attn_modules(model))
        r9.set_ablation(model, [(1, 0)], means)
        abl = model(**x)["logits"]
        r9.set_ablation(model, [], None)
        again = model(**x)["logits"]
    assert torch.allclose(base, again) and not torch.allclose(base, abl)


def test_r9_hawkes_on_tiny_model():
    torch.manual_seed(0)
    edges = bin_edges(48)
    dcfg = {"n_events": 64, "batch_size": 4, "params": {}}
    model = ht.build_model({"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}, 48).eval()
    ab = r9.head_ablation(model, dcfg, edges, [(1, 0)], 6, 0, "cpu", K=300, n_random=2, query_from=24)
    assert "summary" in ab and len(ab["random"]) == 2
    dc = r9.das_controls(model, dcfg, edges, 6, 8, 0, "cpu", blocks=(0, 1), k=3, checkpoints=(16, 32), K=300, steps=2)
    assert len(dc["blocks"]) == 2 and len(dc["alignment"]) == 1 and all(p.requires_grad for p in model.parameters())


# ------------------------------------------------------------------------------------------- FIM-PP (optional)
FIM_YAML = None
for root in (os.environ.get("FIM_SRC", ""), "OpenFIM/src", "../OpenFIM/src"):
    if root and Path(root).exists():
        sys.path.insert(0, str(Path(root).resolve()))
        y = Path(root).resolve().parent / "configs" / "train" / "hawkes" / "david.yaml"
        FIM_YAML = str(y) if y.exists() else None
        break


@pytest.mark.skipif(FIM_YAML is None, reason="OpenFIM source not available")
def test_fimpp_closed_form_matches_its_intensity():
    from mechlab.hawkes import fimpp

    torch.manual_seed(0)
    model = fimpp.random_model(FIM_YAML, "cpu")
    rng = np.random.default_rng(0)
    ctx = np.cumsum(rng.exponential(1.0, (2, 3, 30)), -1)
    inf = np.cumsum(rng.exponential(1.0, (2, 30)), -1)
    edges = bin_edges(48)
    m = 12
    fc = fimpp.forecast(model, ctx, inf[:, :m], edges, "cpu")
    assert np.allclose(fc["P"].sum(-1), 1, atol=1e-6)
    # numerically integrate the model's own intensity after the last event (no later events exist)
    with torch.no_grad():
        x = fimpp._inputs(ctx, inf[:, :m], "cpu")
        grid = inf[:, m - 1:m] + np.linspace(0, 3.0, 3001)[None, :]
        x["intensity_evaluation_times"] = torch.as_tensor(grid[:, None, :], dtype=torch.float32)
        out = model(x)
        lam = out["intensity_function"].evaluate(torch.as_tensor(grid[:, None, :] + 1e-6, dtype=torch.float32))[:, 0, 0].numpy()
    comp = np.concatenate([np.zeros((2, 1)), np.cumsum((lam[:, 1:] + lam[:, :-1]) / 2 * np.diff(grid, axis=1), 1)], 1)
    tau = 3.0
    surv_num = np.exp(-comp[:, -1])
    c = out["intensity_function"].norm_constants.numpy()
    mu, al, be = (fc[k][:, m - 1] * c for k in ("mu", "alpha", "beta"))
    xx = tau / c
    surv_cf = np.exp(-(mu * xx + (al - mu) * (1 - np.exp(-be * xx)) / be))
    assert np.allclose(surv_num, surv_cf, rtol=2e-3, atol=1e-5)
