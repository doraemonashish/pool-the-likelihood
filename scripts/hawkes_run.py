#!/usr/bin/env python
"""In-context Hawkes estimation: train -> evaluate -> probe -> kernel fits -> report.

    python scripts/hawkes_run.py --config configs/hawkes_pilot.yaml
    python scripts/hawkes_run.py --config configs/hawkes_pilot.yaml --smoke

Resumable; finished stages are skipped on re-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mechlab import __version__  # noqa: E402
from mechlab.core.gpt import count_params  # noqa: E402
from mechlab.core.report import md_table, mean_sd, save_fig, verdict_line  # noqa: E402
from mechlab.core.train import train  # noqa: E402
from mechlab.core.utils import deep_update, ensure_dir, get_device, load_config, read_json, seed_everything, write_json  # noqa: E402
from mechlab.hawkes import baselines as bl  # noqa: E402
from mechlab.hawkes import mechanism as mech  # noqa: E402
from mechlab.hawkes import posterior as post  # noqa: E402
from mechlab.hawkes import task  # noqa: E402
from mechlab.hawkes import transplant as tp  # noqa: E402
from mechlab.hawkes.simulate import bin_edges  # noqa: E402

STAGES = ["train", "eval", "probe", "kernel", "mech", "transplant", "baseline", "posterior", "report"]
ALPHA_BINS = {"a0.0-0.3": (0.0, 0.3), "a0.3-0.6": (0.3, 0.6), "a0.6-0.9": (0.6, 0.9), "a0.90-0.98 (unseen, near-critical)": (0.9, 0.98)}


def smoke_overrides(cfg):
    cfg = deep_update(cfg, {"run_name": cfg["run_name"] + "_smoke",
                            "data": {"batch_size": 16, "n_events": 64},
                            "model": {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_mlp": 64},
                            "train": {"steps": 30, "log_every": 10, "ckpt_every": 20, "warmup": 5},
                            "eval": {"n_seqs": 64}, "probe": {"n_seqs": 24, "query_from": 16},
                            "mech": {"n_seqs": 32, "windows": [4]}, "transplant": {"n_seqs": 24, "n_fit": 24},
                            "baseline": {"n_seqs": 12, "n_seqs_powerlaw": 6, "n_prior": 2000},
                            "posterior": {"n_seqs": 12, "n_prior": 1000}})
    keep, seen = [], set()  # the first run of each kernel, so every code path is exercised
    for r in cfg["runs"]:
        k = task.kernel_of(deep_update(cfg["data"], r.get("data", {})))
        if k not in seen:
            seen.add(k)
            keep.append(r)
    cfg["runs"] = keep
    return cfg


def run_one(cfg, run, out_root, stages, device):
    name = run["name"]
    out = ensure_dir(out_root / name)
    dcfg = deep_update(cfg["data"], run.get("data", {}))
    mcfg = deep_update(cfg["model"], run.get("model", {}))
    tcfg = deep_update(cfg["train"], run.get("train", {}))
    seed = int(run.get("seed", cfg.get("seed", 0)))
    seed_everything(seed)
    edges = bin_edges(int(dcfg.get("n_bins", 48)))
    write_json(out / "config.json", {"data": dcfg, "model": mcfg, "train": tcfg, "seed": seed})
    model = task.build_model(mcfg, len(edges) - 1).to(device)
    print(f"\n=== {name}: params={count_params(model):,} null_attn={mcfg.get('null_attn', True)} ===", flush=True)

    def eval_fn(m, step):
        r = task.evaluate(m, dcfg, edges, 512, seed + 1000, device)
        last = [k for k in r if k.endswith("_gap_frac")][-1]
        return {"late_gap_frac": r[last]}

    train(model, task.make_batch_fn(dcfg, seed, edges), task.loss_fn, tcfg, out, device, eval_fn=eval_fn)
    ecfg, pcfg = cfg.get("eval", {}), cfg.get("probe", {})
    if "eval" in stages and not (out / "eval.json").exists():
        res = {"all": task.evaluate(model, dcfg, edges, int(ecfg.get("n_seqs", 4000)), seed + 2000, device)}
        for k, (lo, hi) in ALPHA_BINS.items():
            pr = {**dcfg.get("params", {}), "alpha": (lo, hi)}
            res[k] = task.evaluate(model, dcfg, edges, int(ecfg.get("n_seqs", 4000)) // 2, seed + 2100, device, param_ranges=pr)
        write_json(out / "eval.json", res)
        for k, r in res.items():
            last = [x for x in r if x.endswith("_gap_frac")][-1]
            print(f"[{name}] eval {k}: late-context gap fraction {r[last]:.3f} (0 = oracle, 1 = Poisson)")
    col = None
    kernel_done = (out / "kernel.json").exists() and all("median_r2_pow" in r for r in read_json(out / "kernel.json"))
    if ("probe" in stages and not (out / "probe.json").exists()) or ("kernel" in stages and not kernel_done):
        col = task.collect(model, dcfg, edges, int(pcfg.get("n_seqs", 300)), seed + 3000, device,
                           query_from=int(pcfg.get("query_from", 32)))
    if "probe" in stages and not (out / "probe.json").exists():
        rows = task.probe_params(col)
        write_json(out / "probe.json", rows)
        for r in rows:
            print(f"[{name}] probe layer {r['layer']}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "layer"))
    if "kernel" in stages and not kernel_done:  # round-3 results lack the power-law fit: recompute (cheap)
        rows = task.kernel_fits(col)
        write_json(out / "kernel.json", rows)
        for r in sorted(rows, key=lambda r: -r["median_r2"])[:6]:
            print(f"[{name}] head L{r['layer']}H{r['head']}: exp-fit R2 {r['median_r2']:.2f}, power-fit R2 {r['median_r2_pow']:.2f}, decay {r['median_decay']:.2f} "
                  f"(IQR {r['decay_iqr']:.2f}), corr(decay, beta) {r['spearman_decay_vs_beta']:.2f}, null mass {r['null_mass']:.2f}")
    mc = cfg.get("mech", {})
    if "mech" in stages and not (out / "mech.json").exists():
        nq, qf = int(mc.get("n_seqs", 400)), int(pcfg.get("query_from", 32))
        res = {"intensity": mech.implied_intensity(model, dcfg, edges, nq, seed + 6000, device, query_from=qf),
               "spans": mech.attention_spans(model, dcfg, edges, nq, seed + 6100, device, query_from=qf),
               "windows": mech.window_ablation(model, dcfg, edges, nq, seed + 6200, device,
                                               windows=tuple(mc.get("windows", [4, 16])), query_from=qf)}
        write_json(out / "mech.json", res)
        it = res["intensity"]
        print(f"[{name}] mech implied intensity: Spearman {it['spearman']:.3f}, within-sequence {it['within_seq_spearman_median']:.3f}")
        top = min(res["spans"], key=lambda r: r["spearman_span_vs_beta"])
        print(f"[{name}] mech most beta-adaptive head L{top['layer']}H{top['head']}: corr(span, beta) {top['spearman_span_vs_beta']:.2f}")
        for r in res["windows"]:
            print(f"[{name}] mech layer {r['layer']} limited to last {r['window']} events: +{r['delta_nats']:.4f} nats "
                  f"(low/mid/high beta {r['delta_beta_low']:+.4f}/{r['delta_beta_mid']:+.4f}/{r['delta_beta_high']:+.4f})")
    is_exp = task.kernel_of(dcfg) == "exp"
    tc = cfg.get("transplant", {})
    if "transplant" in stages and not is_exp:
        print(f"[{name}] transplant skipped: implemented for the exponential kernel only")
    if "transplant" in stages and is_exp and not (out / "transplant.json").exists():
        rows = tp.transplant(model, dcfg, edges, int(tc.get("n_seqs", 400)), seed + 8000, device,
                             query_from=int(pcfg.get("query_from", 32)), n_fit=int(tc.get("n_fit", 400)),
                             scales=tuple(tc.get("scales", [1.0])))
        write_json(out / "transplant.json", rows)
        for p_ in tp.PARAMS:
            best = max((r for r in rows if r["param"] == p_ and r["method"] == "param_map"),
                       key=lambda r: r["pattern_own"] - r["pattern_best_other"])
            ctrl = [r for r in rows if r["param"] == p_ and r["layer"] == best["layer"] and r["scale"] == best["scale"] and r["method"] != "param_map"]
            print(f"[{name}] transplant {p_}: best layer {best['layer']} slope {best['slope']:.2f}, own pattern {best['pattern_own']:.2f} "
                  f"vs other parameters' {best['pattern_best_other']:.2f} | "
                  + " | ".join(f"{c['method']}: own {c['pattern_own']:.2f}, other {c['pattern_best_other']:.2f}" for c in ctrl))
    bcfg = cfg.get("baseline", {})
    if "baseline" in stages and not (out / "baseline.json").exists():
        nb = int(bcfg.get("n_seqs", 500) if is_exp else bcfg.get("n_seqs_powerlaw", bcfg.get("n_seqs", 500)))
        res = bl.compare(model, dcfg, edges, nb, seed + 9000, device,
                         checkpoints=tuple(bcfg.get("checkpoints", [2, 4, 8, 16, 32, 64, 128])),
                         n_prior=int(bcfg.get("n_prior", 20000)))
        write_json(out / "baseline.json", res)
        for r in res["rows"]:
            print(f"[{name}] baseline after {r['context']} events: model {r['model']:.3f} | Bayes {r['bayes']:.3f} | "
                  f"ML plug-in {r['ml']:.3f} | oracle {r['oracle']:.3f} | Poisson {r['poisson']:.3f} nats")
    qc = cfg.get("posterior", {})
    if "posterior" in stages and not (out / "posterior.json").exists():
        colp = task.collect(model, dcfg, edges, int(qc.get("n_seqs", pcfg.get("n_seqs", 300))), seed + 3000, device,
                            query_from=int(pcfg.get("query_from", 32)))
        res = post.posterior_probes(colp, dcfg, K=int(qc.get("n_prior", 20000)), seed=seed + 9500,
                                    device="cuda" if torch.cuda.is_available() else "cpu")
        write_json(out / "posterior.json", res)
        for k in res["params"]:
            best = max(res["rows"], key=lambda r: r[f"{k}__posterior_mean"])
            print(f"[{name}] posterior {k}: best layer {best['layer']} R2 true {best[f'{k}__true']:.3f} | posterior mean "
                  f"{best[f'{k}__posterior_mean']:.3f} | posterior sd {best[f'{k}__posterior_sd']:.3f} | ceiling {res['ceiling'][k]:.3f}")


def seed_summary(cfg, out_root) -> list[str]:
    """One row per setting (kernel, null slot, time-aware attention), mean ± sd over seeds."""
    groups: dict = {}
    for run in cfg["runs"]:
        d = out_root / run["name"]
        if not (d / "eval.json").exists():
            continue
        c = read_json(d / "config.json")
        key = (c["data"].get("kernel", "exp"), "null slot" if c["model"].get("null_attn", True) else "no null slot",
               "time-aware" if c["model"].get("pos", "rope") == "rope" else "no time-aware attention")
        a = read_json(d / "eval.json")["all"]
        late = sorted({k.split("_")[0] for k in a}, key=lambda x: int(x[3:].split("-")[0]))[-1]
        m = {"gap": a[f"{late}_gap_frac"], "excess": a[f"{late}_model"] - a[f"{late}_oracle"]}
        if (d / "probe.json").exists():
            pr = read_json(d / "probe.json")
            m["alpha_r2"] = max(r["alpha"] for r in pr)
            m["int_r2"] = max(r["log_intensity"] for r in pr)
        if (d / "mech.json").exists():
            m["int_rho"] = read_json(d / "mech.json")["intensity"]["spearman"]
        if (d / "kernel.json").exists():
            kr = [r for r in read_json(d / "kernel.json") if "median_r2_pow" in r and max(r["median_r2"], r["median_r2_pow"]) >= 0.3]
            if kr:
                m["shape"] = float(np.median([r["median_r2_pow"] - r["median_r2"] for r in kr]))
        if (d / "posterior.json").exists():
            pj = read_json(d / "posterior.json")
            for k, col_ in (("alpha", "post_alpha"), ("log_beta", "post_beta")):
                m[col_] = max(r[f"{k}__posterior_mean"] for r in pj["rows"])
            m["info_alpha"] = max(r["alpha__true"] for r in pj["rows"]) / max(pj["ceiling"]["alpha"], 1e-9)
        if (d / "baseline.json").exists():
            rows = read_json(d / "baseline.json")["rows"]
            m["bayes_gap"] = max(r["model_minus_bayes"] for r in rows if r["context"] >= 8)
        if (d / "transplant.json").exists():
            tj = read_json(d / "transplant.json")
            m["tp_alpha"] = max(r["pattern_own"] - r["pattern_best_other"] for r in tj
                                if r["param"] == "alpha" and r["method"] == "param_map")
        groups.setdefault(key, []).append(m)
    if not groups:
        return []
    cols = [("gap", "gap fraction (late)"), ("excess", "excess nats vs oracle"), ("bayes_gap", "max gap to Bayes (m>=8)"),
            ("alpha_r2", "alpha probe R2"), ("int_r2", "intensity probe R2"), ("int_rho", "implied-intensity Spearman"),
            ("tp_alpha", "alpha edit: own minus other pattern"), ("shape", "head shape: power minus exp R2"),
            ("post_alpha", "posterior-mean R2 (alpha)"), ("post_beta", "posterior-mean R2 (log beta)"),
            ("info_alpha", "alpha: share of available information")]
    rows = [[", ".join(k)] + [mean_sd([m.get(c) for m in ms]) for c, _ in cols] for k, ms in sorted(groups.items())]
    return ["## Summary across seeds", "", "Mean ± sd (number of seeds).", "",
            md_table(["setting"] + [n for _, n in cols], rows), ""]


def report(cfg, out_root):
    th = cfg.get("verdict", {})
    import matplotlib.pyplot as plt

    lines, verdicts = [f"# In-context Hawkes estimation: {cfg['run_name']}", "", f"mechlab {__version__}", ""], []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for run in cfg["runs"]:
        name = run["name"]
        d = out_root / name
        if not (d / "eval.json").exists():
            continue
        ev = read_json(d / "eval.json")
        lines += [f"## {name}", ""]
        if task.kernel_of(deep_update(cfg["data"], run.get("data", {}))) == "powerlaw":
            lines += ["*Power-law kernel* phi(x) = alpha (gamma - 1) / c (1 + x / c)^-gamma: long memory. Here 'beta' means 1 / c "
                      "(the inverse of the kernel's time scale). The parameter transplant is implemented for the exponential "
                      "kernel only, so it is skipped.", ""]
        a = ev["all"]
        ctx = sorted({k.split("_")[0] for k in a}, key=lambda s: int(s[3:].split("-")[0]))
        rows = [[c, a[f"{c}_model"], a[f"{c}_oracle"], a[f"{c}_poisson"], a[f"{c}_gap_frac"]] for c in ctx]
        lines += ["**Next-interval cross-entropy (nats) by number of past events.** gap fraction = (model - oracle) / "
                  "(Poisson - oracle): 0 means the model knows the true parameters, 1 means it learned nothing beyond a rate.", "",
                  md_table(["events in context", "model", "oracle", "in-context Poisson", "gap fraction"], rows), ""]
        late = ctx[-1]
        rows = [[k, ev[k][f"{late}_gap_frac"], ev[k][f"{late}_model"] - ev[k][f"{late}_oracle"]] for k in ALPHA_BINS if k in ev]
        lines += ["**By branching ratio alpha** (late context). The last row was never seen in training. At small alpha "
                  "the oracle is barely better than Poisson, so the gap fraction has a tiny denominator: read the excess nats there.", "",
                  md_table(["alpha range", "gap fraction", "excess nats over oracle"], rows), ""]
        ok = a[f"{late}_gap_frac"] <= th.get("gap_frac", 0.3)
        verdicts.append(verdict_line(f"{name} learns Hawkes in context", ok,
                                     f"late-context gap fraction {a[f'{late}_gap_frac']:.3f} (threshold {th.get('gap_frac', 0.3)})"))
        ax[0].plot(range(len(ctx)), [a[f"{c}_gap_frac"] for c in ctx], marker="o", label=name)
        if (d / "probe.json").exists():
            pr = read_json(d / "probe.json")
            keys = [k for k in pr[0] if k != "layer"]
            lines += ["**Linear probes at query events** (R2, cross-validated by sequence)", "",
                      md_table(["layer"] + keys, [[r["layer"]] + [r[k] for k in keys] for r in pr]), ""]
            best = {k: max(r[k] for r in pr) for k in keys}
            ok = best["alpha"] >= th.get("probe_r2", 0.5) and best["log_beta"] >= th.get("probe_r2", 0.5)
            verdicts.append(verdict_line(f"{name} represents the parameters", ok,
                                         f"best R2 alpha {best['alpha']:.2f}, log beta {best['log_beta']:.2f}, "
                                         f"log intensity {best['log_intensity']:.2f}"))
        if (d / "kernel.json").exists():
            kr = sorted(read_json(d / "kernel.json"), key=lambda r: -r["median_r2"])
            rows = [[f"L{r['layer']}H{r['head']}", r["median_r2"], r.get("median_r2_pow", float("nan")), r["median_decay"], r["decay_iqr"],
                     r["spearman_decay_vs_beta"], r["null_mass"]] for r in kr]
            lines += ["**Attention heads as kernels.** For each head, log(attention to event j / attention to the null slot) is "
                      "regressed on the time gap. High R2 = exponential time profile. corr(decay, beta) > 0 = the head *adapts* "
                      "its decay to the sequence (in-context kernel estimation); high R2 with no correlation = a fixed filter. "
                      "power-fit R2 regresses on log(1 + gap / mean interval) instead: higher than the exp-fit R2 = a long-tailed "
                      "(power-law-like) memory.", "",
                      md_table(["head", "exp-fit R2", "power-fit R2", "median decay", "decay IQR", "corr(decay, true beta)", "null mass"], rows), ""]
            shaped = [r for r in kr if "median_r2_pow" in r and max(r["median_r2"], r["median_r2_pow"]) >= 0.3]
            if shaped:
                diff = float(np.median([r["median_r2_pow"] - r["median_r2"] for r in shaped]))
                verdicts.append(f"- **{name} head profiles (info)**: median power-fit minus exp-fit R2 {diff:+.2f} over {len(shaped)} "
                                f"head(s) with a time profile (R2 >= 0.3 in either fit). Only the difference between exponential "
                                f"and power-law runs is informative: real heads are never exactly either shape.")
            adapt = [r for r in kr if r["median_r2"] >= th.get("kernel_r2", 0.7) and (r["spearman_decay_vs_beta"] or 0) >= th.get("kernel_rho", 0.5)]
            fixed = [r for r in kr if r["median_r2"] >= th.get("kernel_r2", 0.7)]
            verdicts.append(verdict_line(f"{name} has an adaptive kernel head", bool(adapt),
                                         f"{len(adapt)} head(s) with exp-fit R2 >= {th.get('kernel_r2', 0.7)} and corr >= "
                                         f"{th.get('kernel_rho', 0.5)}; {len(fixed)} exponential-looking head(s) in total"))
            ax[1].scatter([r["median_decay"] for r in kr], [r["spearman_decay_vs_beta"] for r in kr],
                          s=[10 + 60 * max(0, r["median_r2"]) for r in kr], alpha=0.6, label=name)
        if (d / "mech.json").exists():
            mj = read_json(d / "mech.json")
            it = mj["intensity"]
            lines += ["**M1. Implied intensity** (from the model's probability of a very short next interval) vs the true "
                      "intensity just after each event. 'Within sequence' asks whether it rises and falls with each burst.", "",
                      md_table(["Spearman (all)", "R2", "slope", "within-seq (median)", "within, low beta", "within, mid beta", "within, high beta"],
                               [[it["spearman"], it["r2"], it["slope"], it["within_seq_spearman_median"], it["within_seq_spearman_beta_low"],
                                 it["within_seq_spearman_beta_mid"], it["within_seq_spearman_beta_high"]]]), ""]
            verdicts.append(verdict_line(f"{name} tracks the intensity in its behaviour",
                                         it["spearman"] >= th.get("intensity_rho", 0.8) and it["within_seq_spearman_median"] >= th.get("within_rho", 0.5),
                                         f"Spearman {it['spearman']:.2f}, within-sequence {it['within_seq_spearman_median']:.2f}"))
            sp = sorted(mj["spans"], key=lambda r: r["spearman_span_vs_beta"])
            rows = [[f"L{r['layer']}H{r['head']}", r["spearman_span_vs_beta"], r["spearman_lag_vs_beta"], r["median_span_time"],
                     r["median_span_events"], r["token_mass"]] for r in sp[:8]]
            lines += ["**M2. Attention spans vs beta** (attention-weighted mean time gap per sequence, correlated with the true "
                      "beta across sequences; negative = the head looks less far back when the kernel decays faster, i.e. it "
                      "adapts its memory, whatever its profile's shape)", "",
                      md_table(["head", "corr(span in time, beta)", "corr(span in events, beta)", "median span (time)",
                                "median span (events)", "mass on events"], rows), ""]
            good = [r for r in sp if r["spearman_span_vs_beta"] <= -th.get("span_rho", 0.5) and r["token_mass"] >= 0.1]
            verdicts.append(verdict_line(f"{name} has a head that adapts its memory to beta", bool(good),
                                         f"strongest: L{sp[0]['layer']}H{sp[0]['head']} corr {sp[0]['spearman_span_vs_beta']:.2f} "
                                         f"(threshold -{th.get('span_rho', 0.5)})"))
            wins = mj["windows"]
            rows = [[r["layer"], r["window"], r["delta_nats"], r["delta_beta_low"], r["delta_beta_mid"], r["delta_beta_high"]] for r in wins]
            lines += ["**M3. Which layers need long context?** One layer at a time may read only the last k events; the "
                      "increase in late-context loss (nats) is split by beta tercile. Large increases at small k mean that layer "
                      "carries long-range information (parameter estimation); near-zero means it only needs recent events.", "",
                      md_table(["layer", "last k events", "loss increase", "low beta", "mid beta", "high beta"], rows), ""]
            kmin = min(r["window"] for r in wins)
            w0 = sorted([r for r in wins if r["window"] == kmin], key=lambda r: -r["delta_nats"])
            verdicts.append(f"- **{name} long-range layer (info)**: limiting layer {w0[0]['layer']} to the last {kmin} events costs "
                            f"+{w0[0]['delta_nats']:.3f} nats; the least sensitive layer ({w0[-1]['layer']}) costs +{w0[-1]['delta_nats']:.3f}")
        if (d / "transplant.json").exists():
            tj = read_json(d / "transplant.json")
            rows = []
            for p_ in tp.PARAMS:
                for l_ in sorted({r["layer"] for r in tj}):
                    for m_ in ("param_map", "ctrl_other_param", "ctrl_random"):
                        cell = [r for r in tj if r["param"] == p_ and r["layer"] == l_ and r["scale"] == 1.0 and r["method"] == m_]
                        if cell:
                            c = cell[0]
                            rows.append([p_, l_, m_, c["slope"], c["pattern_alpha"], c["pattern_log_mu"], c["pattern_log_beta"]])
            lines += ["**M4. Parameter transplant.** After a block, edit the model's estimate of one parameter and compare the "
                      "change in its implied intensity with the true change, had the same history come from the new value. "
                      "slope: 1 = the forecast moves as much as the true process would (not parameter-specific). pattern "
                      "columns: within each sequence, does the change follow that parameter's true-change pattern (alpha: "
                      "after bursts; mu: quiet spells; beta: how fast bursts fade)? An edit is specific if its own column "
                      "wins; the random row shows the model's generic sensitivity profile.", "",
                      md_table(["edited", "after block", "direction", "slope", "pattern alpha", "pattern log mu", "pattern log beta"], rows), ""]
            for p_ in tp.PARAMS:
                cand = [r for r in tj if r["param"] == p_ and r["method"] == "param_map" and r["scale"] == 1.0]
                if not cand:
                    continue
                best = max(cand, key=lambda r: r["pattern_own"] - r["pattern_best_other"])
                rnd = [r for r in tj if r["param"] == p_ and r["layer"] == best["layer"] and r["method"] == "ctrl_random" and r["scale"] == 1.0]
                rnd_pref = (abs(rnd[0]["pattern_own"]) - abs(rnd[0]["pattern_best_other"])) if rnd else 0.0
                margin = best["pattern_own"] - best["pattern_best_other"]
                ok = best["pattern_own"] >= th.get("transplant_corr", 0.5) and best["slope"] >= th.get("transplant_slope", 0.3) \
                    and margin >= 0.2 and margin >= rnd_pref + 0.2
                verdicts.append(verdict_line(f"{name}: the {p_} estimate is causal and specific", ok,
                                             f"after block {best['layer']}: slope {best['slope']:.2f}, own pattern {best['pattern_own']:.2f} "
                                             f"vs other parameters' {best['pattern_best_other']:.2f} (random edit's preference "
                                             f"for this pattern {rnd_pref:+.2f})"))
        if (d / "baseline.json").exists():
            bj = read_json(d / "baseline.json")
            rows = [[r["context"], r["model"], r["bayes"], r["ml"], r["oracle"], r["poisson"],
                     f"{r['model_minus_bayes']:+.4f} ± {r['model_minus_bayes_se']:.4f}",
                     (f"{r['model_minus_ml']:+.4f} ± {r['model_minus_ml_se']:.4f}" if "model_minus_ml_se" in r else f"{r['model_minus_ml']:+.4f}"),
                     int(round(r["median_ess"]))] for r in bj["rows"]]
            lines += ["**M5. Against the best possible forecaster.** Cross-entropy (nats) of the next interval after m events. "
                      "Bayes = the best possible forecast given only the history and the training prior (Bayes-optimal, importance "
                      "sampling over prior draws); ML = fit the parameters by maximum likelihood (the best of the same draws), then "
                      "forecast; oracle = knows the true parameters (unattainable). "
                      "model - Bayes near 0 means the transformer is close to Bayes-optimal; below ML at small m means it beats "
                      "the classical fit-then-forecast approach.", "",
                      md_table(["events seen m", "model", "Bayes", "ML plug-in", "oracle", "Poisson", "model - Bayes (± s.e.)",
                                "model - ML (± s.e.)", "effective prior samples"], rows), ""]
            gaps = [r["model_minus_bayes"] for r in bj["rows"] if r["context"] >= 8]
            # "beats" = better by more than two standard errors of the paired difference
            beats_ml = [r for r in bj["rows"] if r["model_minus_ml"] < -2 * r.get("model_minus_ml_se", 0.0)]
            if gaps:
                verdicts.append(verdict_line(f"{name} is close to Bayes-optimal", max(gaps) <= th.get("bayes_gap", 0.02),
                                             f"largest gap to Bayes from 8 events on: {max(gaps):+.4f} nats (threshold "
                                             f"{th.get('bayes_gap', 0.02)}); beats the ML plug-in (by > 2 s.e.) at m = "
                                             f"{', '.join(str(r['context']) for r in beats_ml) or 'none'}"))
        if (d / "posterior.json").exists():
            pj = read_json(d / "posterior.json")
            pn = pj["params"]
            rows = [[r["layer"]] + [r[f"{k}__{t}"] for k in pn for t in ("true", "posterior_mean", "posterior_sd")] for r in pj["rows"]]
            rows.append(["ceiling"] + [x for k in pn for x in (pj["ceiling"][k], "", "")])
            lines += ["**P1. Parameters or posterior?** Probe R2 at query events (cross-validated by sequence) for the true "
                      "parameter, its posterior mean given the events so far, and its posterior standard deviation "
                      f"(importance sampling, {pj['K']} prior draws, median effective sample size {pj['median_ess']:.0f}). "
                      "ceiling = R2 of the posterior mean itself as a predictor of the true parameter: the most any function of "
                      "the history can explain. A model that holds the posterior scores near 1 on 'posterior mean' and near the "
                      "ceiling on 'true'.", "",
                      md_table(["layer"] + [f"{k}: {t}" for k in pn for t in ("true", "posterior mean", "posterior sd")], rows), ""]
            for k in pn:
                best = max(pj["rows"], key=lambda r: r[f"{k}__posterior_mean"])
                bt = max(r[f"{k}__true"] for r in pj["rows"])
                bp, ce = best[f"{k}__posterior_mean"], pj["ceiling"][k]
                verdicts.append(verdict_line(f"{name} encodes the posterior mean of {k}",
                                             bp >= th.get("posterior_r2", 0.85) and bp >= bt + th.get("posterior_margin", 0.05),
                                             f"best R2 for the posterior mean {bp:.3f} (layer {best['layer']}) vs {bt:.3f} for the true "
                                             f"value (thresholds {th.get('posterior_r2', 0.85)}, margin {th.get('posterior_margin', 0.05)})"))
                verdicts.append(verdict_line(f"{name} holds the available information about {k}",
                                             bt >= th.get("info_share", 0.9) * ce,
                                             f"true-value R2 {bt:.3f} = {bt / max(ce, 1e-9):.0%} of the ceiling {ce:.3f} "
                                             f"(threshold {th.get('info_share', 0.9):.0%}); posterior sd R2 "
                                             f"{max(r[f'{k}__posterior_sd'] for r in pj['rows']):.3f}"))
    ax[0].set_xlabel("context bucket"); ax[0].set_ylabel("gap fraction"); ax[0].set_title("In-context learning curve")
    ax[1].set_xscale("log"); ax[1].set_xlabel("median decay (1/time)"); ax[1].set_ylabel("corr(decay, beta)")
    ax[1].set_title("Heads (size = exp-fit R2)")
    for x in ax:
        if x.lines or x.collections:
            x.legend(fontsize=7)
    fig_path = save_fig(fig, out_root / "report" / "summary.png")
    text = "\n".join(lines[:4] + seed_summary(cfg, out_root) + ["## Verdict", ""] + verdicts
                     + ["", f"![summary]({Path(fig_path).name})", ""] + lines[4:])
    (out_root / "report" / "report.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = smoke_overrides(cfg)
    out_root = ensure_dir(Path(cfg.get("out_root", "outputs")) / cfg["run_name"])
    device = get_device(cfg.get("device", "auto"))
    print(f"mechlab {__version__} | torch {torch.__version__} | device {device}")
    for run in cfg["runs"]:
        if not args.only or run["name"] in args.only:
            run_one(cfg, run, out_root, args.stages, device)
    if "report" in args.stages:
        report(cfg, out_root)
        print(f"\nreport: {out_root / 'report' / 'report.md'}")


if __name__ == "__main__":
    main()
