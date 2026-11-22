import numpy as np
import torch
from scipy import stats

from mechlab.hawkes import task
from mechlab.hawkes.simulate import bin_edges, oracle_bin_probs, sample_params, simulate, to_bins


def test_time_rescaling_theorem():
    rng = np.random.default_rng(0)
    p = sample_params(rng, 1000)
    s = simulate(p["mu"], p["alpha"], p["beta"], 60, rng)
    E_prev = np.concatenate([np.zeros((1000, 1)), s["E_after"][:, :-1]], 1)
    tau, b = s["intervals"], p["beta"][:, None]
    Lam = p["mu"][:, None] * tau + (E_prev / b) * (1 - np.exp(-b * tau))
    assert stats.kstest(Lam.ravel(), "expon").pvalue > 0.01


def test_poisson_limit_rate():
    rng = np.random.default_rng(1)
    s = simulate(np.full(2000, 2.0), np.zeros(2000), np.ones(2000), 50, rng)
    assert abs(s["intervals"].mean() - 0.5) < 0.02


def test_oracle_probs_normalised_and_bins():
    edges = bin_edges(48)
    P = oracle_bin_probs(np.array([1.0, 0.5]), np.array([0.0, 3.0]), np.array([2.0, 5.0]), edges)
    assert np.allclose(P.sum(-1), 1)
    assert to_bins(np.array([1e-9, 1.0, 1e9]), edges).tolist() == [0, to_bins(np.array([1.0]), edges)[0], 47]


def test_pipeline_runs():
    edges = bin_edges(48)
    d = {"batch_size": 6, "n_events": 40}
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    b = {k: torch.from_numpy(v) for k, v in task.batch_for_step(0, d, 0, edges).items()}
    loss, _ = task.loss_fn(m, b)
    assert torch.isfinite(loss)
    ev = task.evaluate(m, d, edges, 12, 0, torch.device("cpu"))
    assert all(np.isfinite(v) for v in ev.values())
    col = task.collect(m, d, edges, 8, 0, torch.device("cpu"), query_from=10)
    assert len(task.probe_params(col)) == 3
    assert task.kernel_fits(col, max_queries=4)


def test_mechanism_runs_and_unlimited_window_changes_nothing():
    from mechlab.hawkes import mechanism as mech
    edges = bin_edges(48)
    d = {"batch_size": 6, "n_events": 40}
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    dev = torch.device("cpu")
    it = mech.implied_intensity(m, d, edges, 12, 0, dev, query_from=10)
    assert -1 <= it["spearman"] <= 1
    sp = mech.attention_spans(m, d, edges, 12, 0, dev, query_from=10)
    assert len(sp) == 4 and all(0 <= r["token_mass"] <= 1.0001 for r in sp)
    w = mech.window_ablation(m, d, edges, 12, 0, dev, windows=(1000,), query_from=10)
    assert all(abs(r["delta_nats"]) < 1e-5 for r in w)
    w2 = mech.window_ablation(m, d, edges, 12, 0, dev, windows=(1,), query_from=10)
    assert any(abs(r["delta_nats"]) > 1e-6 for r in w2)


def test_variants_without_null_slot_or_time_rotary():
    edges = bin_edges(48)
    d = {"batch_size": 6, "n_events": 40}
    for mc in ({"null_attn": False}, {"pos": "none"}):
        m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64, **mc}, len(edges) - 1)
        col = task.collect(m, d, edges, 6, 0, torch.device("cpu"), query_from=10)
        assert task.kernel_fits(col, max_queries=4)


def test_excitation_matches_simulator_and_transplant_runs():
    from mechlab.hawkes import transplant as tp
    rng = np.random.default_rng(4)
    p = sample_params(rng, 50)
    s = simulate(p["mu"], p["alpha"], p["beta"], 40, rng)
    assert np.allclose(tp.excitation_after(s["times"], p["alpha"], p["beta"]), s["E_after"])
    edges = bin_edges(48)
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    rows = tp.transplant(m, {"batch_size": 6, "n_events": 40}, edges, 10, 0, torch.device("cpu"), query_from=10, n_fit=10)
    assert len(rows) == 3 * 2 * 3  # params x layers x methods
    assert all(np.isfinite(r["slope"]) for r in rows)


def test_powerlaw_time_rescaling_and_oracle():
    from mechlab.hawkes.simulate import oracle_bin_probs_powerlaw, poisson_bin_probs, sample_params_powerlaw, simulate_powerlaw
    rng = np.random.default_rng(7)
    B, N = 400, 40
    p = sample_params_powerlaw(rng, B)
    s = simulate_powerlaw(p["mu"], p["alpha"], p["c"], p["gamma"], N, rng)
    t = s["times"]
    a, c, g = p["alpha"][:, None], p["c"][:, None], p["gamma"][:, None]
    Phi = lambda x: a * (1 - (1 + x / c) ** (1 - g))  # noqa: E731
    Lam = np.zeros((B, N))
    for i in range(N):
        prev = t[:, i - 1] if i else np.zeros(B)
        Lam[:, i] = p["mu"] * (t[:, i] - prev)
        if i:
            past = t[:, :i]
            Lam[:, i] += (Phi(t[:, i:i + 1] - past) - Phi(prev[:, None] - past)).sum(1)
    assert stats.kstest(Lam.ravel(), "expon").pvalue > 0.01
    edges = bin_edges(48)
    P = oracle_bin_probs_powerlaw(t, p["mu"], p["alpha"], p["c"], p["gamma"], edges)
    assert P.shape == (B, N, 47 + 1) and np.allclose(P.sum(-1), 1)
    y = to_bins(s["intervals"][:, 1:], edges)
    ce_oracle = -np.log(np.take_along_axis(P[:, :-1], y[..., None], -1)).mean()
    rate = np.arange(1, N) / t[:, :-1]
    Pp = poisson_bin_probs(rate, edges)
    ce_pois = -np.log(np.take_along_axis(Pp, y[..., None], -1)).mean()
    assert ce_oracle < ce_pois


def test_powerlaw_pipeline_runs():
    edges = bin_edges(48)
    d = {"batch_size": 4, "n_events": 30, "kernel": "powerlaw",
         "params": {"mu": [0.2, 2.0], "alpha": [0.0, 0.9], "c": [0.1, 2.0], "gamma": [1.5, 3.0]}}
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    b = {k: torch.from_numpy(v) for k, v in task.batch_for_step(0, d, 0, edges).items()}
    loss, _ = task.loss_fn(m, b)
    assert torch.isfinite(loss)
    ev = task.evaluate(m, d, edges, 8, 0, torch.device("cpu"))
    assert all(np.isfinite(v) for v in ev.values())
    col = task.collect(m, d, edges, 6, 0, torch.device("cpu"), query_from=10)
    assert {"alpha", "gamma", "log_intensity"} <= set(col["targets"])


def test_bayes_baseline_ordering():
    from mechlab.hawkes.baselines import compare
    edges = bin_edges(48)
    d = {"batch_size": 4, "n_events": 40}
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    out = compare(m, d, edges, 200, 0, torch.device("cpu"), checkpoints=(2, 8, 32), n_prior=3000)
    rows = {r["context"]: r for r in out["rows"]}
    assert set(rows) == {2, 8, 32}
    for r in rows.values():
        assert r["oracle"] < r["bayes"] + 0.05  # the oracle knows the parameters
        assert r["bayes"] < r["poisson"]        # the Bayes forecaster uses the excitation
        assert r["bayes"] <= r["ml"] + 0.05     # averaging beats plugging in one estimate, especially early
    assert rows[2]["ml"] - rows[2]["bayes"] > rows[32]["ml"] - rows[32]["bayes"] - 0.05


def test_kernel_fits_tell_exponential_from_power_law_heads():
    rng = np.random.default_rng(3)
    B, N = 8, 60
    times = np.cumsum(rng.exponential(1.0, (B, N)), 1)
    beta = rng.uniform(0.5, 3.0, B)
    A = np.zeros((B, 2, N, N + 1))
    for i in range(N):
        dt = times[:, i:i + 1] - times[:, : i + 1]
        A[:, 0, i, : i + 1] = np.exp(-0.5 * dt)
        A[:, 1, i, : i + 1] = (1 + dt / 0.3) ** -1.5
        A[:, :, i, -1] = 1.0
    A /= A.sum(-1, keepdims=True)
    col = {"times": times, "attn": [A], "query_idx": np.arange(20, N), "beta": beta}
    rows = {r["head"]: r for r in task.kernel_fits(col, max_queries=6)}
    assert rows[0]["median_r2"] > 0.99 and rows[0]["median_r2"] > rows[0]["median_r2_pow"]
    assert rows[1]["median_r2_pow"] > rows[1]["median_r2"]


def test_likelihoods_match_direct_computation():
    from mechlab.hawkes.baselines import _loglik_checkpoints, bin_probs_powerlaw_torch
    from mechlab.hawkes.posterior import loglik_exp, loglik_powerlaw, prior_draws
    from mechlab.hawkes.simulate import oracle_bin_probs_powerlaw, sample_params_powerlaw, simulate_powerlaw
    rng = np.random.default_rng(2)
    p = sample_params(rng, 1)
    t = simulate(p["mu"], p["alpha"], p["beta"], 30, rng)["times"][0]
    pr = prior_draws({}, 50, 1)
    ll = loglik_exp(t, pr, 30)
    ref, _ = _loglik_checkpoints(t, pr["mu"], pr["alpha"], pr["beta"], [5, 30])
    assert np.allclose(ll[:, 4], ref[5]) and np.allclose(ll[:, 29], ref[30])
    dpl = {"kernel": "powerlaw"}
    q = sample_params_powerlaw(rng, 1)
    t = simulate_powerlaw(q["mu"], q["alpha"], q["c"], q["gamma"], 25, rng)["times"][0]
    pr = prior_draws(dpl, 7, 3)
    ll = loglik_powerlaw(t, pr, 25, "cpu")
    for k in range(7):  # brute force
        mu, a, c, g = (pr[x][k] for x in ("mu", "alpha", "c", "gamma"))
        lam = [mu + sum(a * (g - 1) / c * (1 + (t[i] - t[j]) / c) ** -g for j in range(i)) for i in range(25)]
        comp = mu * t[24] + sum(a * (1 - (1 + (t[24] - t[j]) / c) ** (1 - g)) for j in range(24))
        assert np.isclose(ll[k, 24], np.sum(np.log(lam)) - comp)
    edges = bin_edges(48)
    P = bin_probs_powerlaw_torch(t[:10], {k: q[k] for k in ("mu", "alpha", "c", "gamma")}, edges, "cpu")[0]
    O = oracle_bin_probs_powerlaw(t[None, :10], q["mu"], q["alpha"], q["c"], q["gamma"], edges)[0, 9]
    assert np.allclose(P, O, atol=1e-9)


def test_posterior_is_calibrated_and_probes_run():
    from mechlab.hawkes.posterior import posterior_probes, posterior_stats
    rng = np.random.default_rng(5)
    p = sample_params(rng, 150)
    s = simulate(p["mu"], p["alpha"], p["beta"], 40, rng)
    pos = np.array([4, 19, 39])
    post = posterior_stats({}, s["times"], pos, K=4000, seed=9)
    for name, truth in (("alpha", p["alpha"]), ("log_beta", np.log(p["beta"]))):
        err2 = ((truth[:, None] - post[name]["mean"]) ** 2).mean(0)
        var = (post[name]["sd"] ** 2).mean(0)
        assert np.all(np.abs(err2 / var - 1) < 0.35)          # E[(theta - mean)^2] = E[posterior variance]
        assert err2[-1] < err2[0]                              # more events, less uncertainty
    edges = bin_edges(48)
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    col = task.collect(m, {"batch_size": 6, "n_events": 30}, edges, 10, 0, torch.device("cpu"), query_from=10)
    res = posterior_probes(col, {"n_events": 30}, K=500)
    assert len(res["rows"]) == 3 and {"alpha__posterior_mean", "log_beta__posterior_sd"} <= set(res["rows"][0])
    assert 0 < res["ceiling"]["alpha"] < 1


def test_powerlaw_bayes_baseline_ordering():
    from mechlab.hawkes.baselines import compare
    edges = bin_edges(48)
    d = {"batch_size": 4, "n_events": 40, "kernel": "powerlaw"}
    m = task.build_model({"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64}, len(edges) - 1)
    out = compare(m, d, edges, 80, 0, "cpu", checkpoints=(2, 16), n_prior=1500)
    for r in out["rows"]:
        assert r["oracle"] < r["bayes"] + 0.05 and r["bayes"] < r["poisson"] and r["bayes"] <= r["ml"] + 0.05
