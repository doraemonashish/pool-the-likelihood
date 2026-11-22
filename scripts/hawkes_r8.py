#!/usr/bin/env python
"""Hawkes round 8: pooled traces (P1-P4), interchange with DAS (I2-I3), misspecification confidence intervals (M1).
Uses the trained models of configs/hawkes_main.yaml; writes r8_*.json next to them and a report.

    python scripts/hawkes_r8.py --config configs/hawkes_r8.yaml
    python scripts/hawkes_r8.py --config configs/hawkes_r8.yaml --smoke     # tiny models, trained here
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mechlab import __version__  # noqa: E402
from mechlab.core.report import md_table, verdict_line  # noqa: E402
from mechlab.core.train import train  # noqa: E402
from mechlab.core.utils import deep_update, ensure_dir, get_device, load_config, read_json, write_json  # noqa: E402
from mechlab.hawkes import r7, r8  # noqa: E402
from mechlab.hawkes import task  # noqa: E402
from mechlab.hawkes.simulate import bin_edges  # noqa: E402

STAGES = ["misspec_ci", "pooled", "das", "report"]


def run_cfgs(main: dict):
    for run in main["runs"]:
        yield run["name"], deep_update(main["data"], run.get("data", {})), deep_update(main["model"], run.get("model", {}))


def load_model(out: Path, dcfg, mcfg, edges, device, smoke_steps: int = 0):
    model = task.build_model(mcfg, len(edges) - 1).to(device)
    if not (out / "model.pt").exists():
        if not smoke_steps:
            return None
        train(model, task.make_batch_fn({**dcfg, "batch_size": 8}, 0, edges), task.loss_fn,
              {"steps": smoke_steps, "log_every": 10, "warmup": 2, "ckpt_every": 1000}, out, device, verbose=False)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    model.eval()
    return model


def get_bayes(paths, fams, datasets, cps, edges, K, device, save_to: Path) -> dict:
    """Reuse the round-7 cache when its shapes match these datasets; otherwise compute and cache."""
    S = {f: len(b["times64"]) for f, b in datasets.items()}
    for p in paths:
        if p.exists():
            z = np.load(p, allow_pickle=True)["bayes"].item()
            if all(v["P"].shape[:2] == (S[df], len(cps)) for (pf, df), v in z.items()):
                print(f"[bayes] reusing {p}", flush=True)
                return z
    out = {}
    for pf, pd in fams.items():
        for df, b in datasets.items():
            print(f"[bayes] prior {pf} on {df} data ({len(b['times64'])} sequences)", flush=True)
            out[(pf, df)] = r7.bayes_dists(pd, b["times64"], cps, edges, K, 4242, device)
    np.savez(save_to, bayes=np.array(out, dtype=object))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stages", nargs="*", default=STAGES)
    args = ap.parse_args()
    cfg = load_config(args.config)
    main_cfg = load_config(cfg["main_config"])
    smoke_steps = 0
    r7_name = cfg.get("r7_run_name", "hawkes_r7")
    if args.smoke:
        main_cfg = deep_update(main_cfg, {"run_name": main_cfg["run_name"] + "_r8smoke", "data": {"n_events": 64},
                                          "model": {"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}})
        keep, seen = [], set()
        for r in main_cfg["runs"]:
            k = task.kernel_of(deep_update(main_cfg["data"], r.get("data", {})))
            if k not in seen:
                seen.add(k)
                keep.append(r)
        main_cfg["runs"] = keep
        cfg = deep_update(cfg, {"misspec_ci": {"n_seqs": 6, "K": 300, "checkpoints": [8, 32], "n_boot": 50},
                                "pooled": {"n_seqs": 10, "n_test": 6, "K": 300, "query_from": 24, "mlp_epochs": 5},
                                "das": {"n_fit": 8, "n_pairs": 8, "K": 300, "checkpoints": [16, 32], "steps": 3, "ks": [3]}})
        smoke_steps = 20
        r7_name = "none"
    base = Path(main_cfg.get("out_root", "outputs"))
    out_root = ensure_dir(base / main_cfg["run_name"])
    rep_dir = ensure_dir(base / (cfg.get("run_name", "hawkes_r8") + ("_smoke" if args.smoke else "")))
    device = get_device("auto")
    print(f"mechlab {__version__} | torch {torch.__version__} | device {device}")
    edges = bin_edges(int(main_cfg["data"].get("n_bins", 48)))
    runs = [(n, d, m) for n, d, m in run_cfgs(main_cfg) if not args.only or n in args.only]
    fams = {}
    for n, d, m in run_cfgs(main_cfg):
        fams.setdefault(task.kernel_of(d), {k: v for k, v in d.items() if k != "batch_size"})
    mc, pc, dc = cfg.get("misspec_ci", {}), cfg.get("pooled", {}), cfg.get("das", {})

    if "misspec_ci" in args.stages and len(fams) == 2:
        cps = [m for m in (int(x) for x in mc.get("checkpoints", [8, 16, 32, 64, 128])) if m <= int(main_cfg["data"]["n_events"]) - 1]
        # the same sequences as round 7 (same seeds and sizes), so the round-7 Bayes cache can be reused
        datasets = {f: task.make_batch({**d, "batch_size": int(mc.get("n_seqs", 200))}, [77, 7 if f == "exp" else 8], edges,
                                       with_truth=True) for f, d in fams.items()}
        bayes = get_bayes([base / r7_name / "bayes_cache.npz", rep_dir / "bayes_cache.npz"], fams, datasets, cps, edges,
                          int(mc.get("K", 20000)), device, rep_dir / "bayes_cache.npz")
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if (out / "r8_misspec_ci.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            res = r8.misspec_ci(model, dcfg, datasets, bayes, edges, cps, device, n_boot=int(mc.get("n_boot", 2000)))
            write_json(out / "r8_misspec_ci.json", res)
            for fam, r in res["data"].items():
                if r["misspecified"]:
                    print(f"[{name}] on {fam} data: KL own {r['kl_own']:.4f} true {r['kl_true']:.4f} | diff CI "
                          f"[{r['diff_ci95'][0]:.4f}, {r['diff_ci95'][1]:.4f}] | ratio {r['ratio_own_over_true']:.3f} "
                          f"[{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}] | closer to own {r['share_seqs_closer_to_own']:.2f}",
                          flush=True)
    if "pooled" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if task.kernel_of(dcfg) != "exp" or (out / "r8_pooled.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            res = r8.pooled_tests(model, dcfg, edges, int(pc.get("n_seqs", 400)), 31, device, K=int(pc.get("K", 20000)),
                                  n_test=int(pc.get("n_test", 400)), query_from=int(pc.get("query_from", 48)),
                                  mlp_epochs=int(pc.get("mlp_epochs", 200)))
            write_json(out / "r8_pooled.json", res)
            for k, v in res["sufficiency"].items():
                print(f"[{name}] {k}: MLP R2 count+time {v['count_time_mlp']:.3f} | bank {v['bank_mlp']:.3f} | pooled "
                      f"{v['pooled_mlp']:.3f} | bank+pooled {v['bank_pooled_mlp']:.3f} | residual probe {v['residual_best_linear']:.3f}",
                      flush=True)
            print(f"[{name}] pooled-trace R2 by layer: " + " ".join(f"{r['pooled_r2_mean']:.2f}" for r in res["representation"]))
            for r in res["reconstruction"]:
                print(f"[{name}] reconstruction after block {r['block']}: recovered bank {r['recovered_bank']} | "
                      f"bank+pooled {r['recovered_bank_pooled']}", flush=True)
    if "das" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if task.kernel_of(dcfg) != "exp" or (out / "r8_das.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            res = r8.interchange_das(model, dcfg, edges, int(dc.get("n_fit", 400)), int(dc.get("n_pairs", 600)), 21, device,
                                     checkpoints=tuple(dc.get("checkpoints", [64, 128])), K=int(dc.get("K", 20000)),
                                     ks=tuple(dc.get("ks", [3, 8])), steps=int(dc.get("steps", 300)))
            write_json(out / "r8_das.json", res)
            for r in res["blocks"]:
                print(f"[{name}] block {r['block']} {r['mode']}: " + " ".join(
                    f"{k} {r[k]:.3f}" for k in r if k.startswith(("decoder", "encoder", "full", "random", "das"))
                    and not k.endswith(("train", "expected")) and "overlap" not in k), flush=True)
    if "report" in args.stages:
        print(report(cfg, main_cfg, out_root, rep_dir))


def report(cfg, main_cfg, out_root, rep_dir) -> str:
    th = cfg.get("verdict", {})
    lines, v = ["# Hawkes round 8: pooled traces, interchange with DAS, misspecification intervals", "", f"mechlab {__version__}", ""], []
    # ---- M1 intervals
    rows, ok = [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r8_misspec_ci.json"
        if not p.exists():
            continue
        r = read_json(p)
        for fam, d in r["data"].items():
            if not d["misspecified"]:
                continue
            rows.append([name, r["own_family"], fam, f"{d['kl_own']:.4f}", f"{d['kl_true']:.4f}",
                         f"[{d['diff_ci95'][0]:.4f}, {d['diff_ci95'][1]:.4f}]", f"{d['ratio_own_over_true']:.3f}",
                         f"[{d['ratio_ci95'][0]:.3f}, {d['ratio_ci95'][1]:.3f}]", f"{d['share_seqs_closer_to_own']:.2f}",
                         " ".join(f"{x:.0f}" for x in d["ess_median_true_prior"])])
            ok.append(d["diff_ci95"][0] > 0 and d["ratio_ci95"][1] < th.get("m1_ratio", 0.5))
    if rows:
        lines += ["## M1 with confidence intervals", "", "Per-sequence KL to the Bayes forecast under the model's own prior and under the "
                  "true family's prior, averaged over context lengths; 95% bootstrap intervals over sequences; ESS = median "
                  "importance-sampling effective sample size of the true-family Bayes reference, by context length.", "",
                  md_table(["run", "trained on", "data", "KL own", "KL true", "true - own (95% CI)", "own / true", "95% CI",
                            "share closer to own", "ESS (true prior)"], rows), ""]
        v.append(verdict_line("M1 holds with confidence intervals", all(ok),
                              f"{sum(ok)}/{len(ok)} models: true - own > 0 and own/true < {th.get('m1_ratio', 0.5)} at the 95% level"))
    # ---- pooled traces
    rows, rep_rows, rec_rows, att_rows = [], [], [], []
    p1, p2, p4, n_pool_heads = [], [], [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r8_pooled.json"
        if not p.exists():
            continue
        r = read_json(p)
        s = r["sufficiency"]
        rows.append([name] + [f"{s[k]['bank_mlp']:.3f} / {s[k]['bank_pooled_mlp']:.3f} / {s[k]['residual_best_linear']:.3f}"
                              for k in r8.PARAMS])
        p1.append(all(s[k]["bank_pooled_mlp"] >= s[k]["residual_best_linear"] - th.get("p1_tol", 0.05) for k in r8.PARAMS))
        rp = [x["pooled_r2_mean"] for x in r["representation"]]
        rep_rows.append([name] + [f"{x:.2f}" for x in rp] + [" ".join(f"{x['bank_r2_mean']:.2f}" for x in r["representation"])])
        p2.append(max(rp) >= th.get("p2_r2", 0.8) and max(rp[2:]) - rp[1] >= th.get("p2_gain", 0.1))
        rec = [x for x in r["reconstruction"] if x["recovered_bank"] is not None and x["recovered_bank_pooled"] is not None]
        rec_rows.append([name] + [f"block {x['block']}: {x['recovered_bank']:.2f} -> {x['recovered_bank_pooled']:.2f}" for x in r["reconstruction"]
                                  if x["recovered_bank"] is not None and x["recovered_bank_pooled"] is not None])
        p4.append(any(x["block"] >= 1 and x["recovered_bank_pooled"] - x["recovered_bank"] >= th.get("p4_gain", 0.1) for x in rec))
        heads = [a for a in r["attention"] if a["block"] >= 1]
        pool = [a for a in heads if a["tv_uniform"] < th.get("p3_tv", 0.25)]
        n_pool_heads.append(len(pool))
        att_rows.append([name, f"{len(pool)}/{len(heads)}",
                         " ".join(f"b{a['block']}h{a['head']}:{a['tv_uniform']:.2f}" for a in sorted(r["attention"], key=lambda a: a["tv_uniform"])[:5])])
    if rows:
        lines += ["## P: pooled traces", "", "Posterior-mean R2 from features (MLP): current decayed counts (bank) / bank + pooled "
                  "log-traces / best linear probe on the residual stream.", "",
                  md_table(["run", "log mu", "alpha", "log beta"], rows), "",
                  "Pooled traces decoded from the residual stream (mean R2 over 96 pooled features), by layer (0 = embeddings); last "
                  "column: the current decayed counts.", "",
                  md_table(["run"] + [f"L{i}" for i in range(len(rep_rows[0]) - 2)] + ["bank R2 by layer"], rep_rows), "",
                  "Reconstruction: share of the count+time damage recovered by a linear map of [bank] -> [bank + pooled].", "",
                  md_table(["run"] + [f"" for _ in range(max(len(x) for x in rec_rows) - 1)], [x + [""] * (max(len(y) for y in rec_rows) - len(x)) for x in rec_rows]), "",
                  "Attention: heads in blocks >= 1 whose weight over past events is close to uniform (total variation < "
                  f"{th.get('p3_tv', 0.25)}), and the five most uniform heads.", "",
                  md_table(["run", "pooling heads", "most uniform heads (TV)"], att_rows), ""]
        v.append(verdict_line("P1 pooled traces suffice for the posterior", sum(p1) >= 0.75 * len(p1),
                              f"{sum(p1)}/{len(p1)} models within {th.get('p1_tol', 0.05)} of the residual probe for all three"))
        v.append(verdict_line("P2 the residual stream encodes pooled traces, more after block 0", sum(p2) >= 0.75 * len(p2),
                              f"{sum(p2)}/{len(p2)} models"))
        v.append(f"- **P3 pooling heads (exploratory)**: {', '.join(str(x) for x in n_pool_heads)} heads per model in blocks >= 1")
        v.append(verdict_line("P4 pooled traces explain later blocks", sum(p4) >= 0.75 * len(p4), f"{sum(p4)}/{len(p4)} models"))
    # ---- DAS
    rows, i2, i3 = [], [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r8_das.json"
        if not p.exists():
            continue
        r = read_json(p)
        kmax = max(int(k[3:]) for k in r["blocks"][0] if k.startswith("das") and k[3:].isdigit())
        for b in r["blocks"]:
            rows.append([name, b["block"], b["mode"], f"{b['decoder3']:.3f}", f"{b['encoder3']:.3f}", f"{b.get('das3', float('nan')):.3f}",
                         f"{b[f'das{kmax}']:.3f}", f"{b[f'random{kmax}']:.3f}", f"{b['full']:.3f}",
                         f"{b[f'das{kmax}_overlap_decoder']:.2f} (chance {b[f'das{kmax}_overlap_random_expected']:.2f})"])
        best = max(r["blocks"], key=lambda b: b[f"das{kmax}"])
        i2.append(best[f"das{kmax}"] >= th.get("i2_transfer", 0.5) and best[f"das{kmax}"] >= th.get("i2_ratio", 3.0) * max(best[f"random{kmax}"], 0.01))
        if i2[-1]:
            i3.append(best[f"das{kmax}_overlap_decoder"] >= th.get("i3_overlap", 0.3))
    if rows:
        lines += ["## I2: interchange by block, with DAS", "", "Transfer on held-out pairs (1 = the patched forecast equals the Bayes "
                  "forecast with A's posterior on B's history). query = patch the last position only; prefix = every position up to it. "
                  "full = swap the whole residual. Overlap = share of the 3 probe directions inside the DAS subspace.", "",
                  md_table(["run", "block", "mode", "decoder (3)", "encoder (3)", "DAS (3)", f"DAS ({kmax})", f"random ({kmax})", "full",
                            "DAS overlap with probe"], rows), ""]
        v.append(verdict_line("I2 a learned subspace transplants the posterior", sum(i2) >= 0.75 * len(i2),
                              f"{sum(i2)}/{len(i2)} models reach transfer >= {th.get('i2_transfer', 0.5)} and >= {th.get('i2_ratio', 3.0)}x random"))
        if i3:
            v.append(verdict_line("I3 that subspace is the probe's posterior subspace", sum(i3) >= 0.75 * len(i3), f"{sum(i3)}/{len(i3)} models"))
    text = "\n".join(lines[:4] + ["## Verdict", ""] + (v or ["(no results yet)"]) + [""] + lines[4:])
    (rep_dir / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    main()
