#!/usr/bin/env python
"""Hawkes round 7: misspecification (M), posterior interchange (I), kernel bank (K). Uses the trained models of
configs/hawkes_main.yaml (outputs/hawkes_main/<run>/model.pt); writes r7_*.json next to them and a report.

    python scripts/hawkes_r7.py --config configs/hawkes_r7.yaml
    python scripts/hawkes_r7.py --config configs/hawkes_r7.yaml --smoke     # tiny models, trained here
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
from mechlab.hawkes import r7  # noqa: E402
from mechlab.hawkes import task  # noqa: E402
from mechlab.hawkes.simulate import bin_edges  # noqa: E402

STAGES = ["misspec", "interchange", "bank", "report"]


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


def best_posterior_layer(out: Path, default: int) -> int:
    if (out / "posterior.json").exists():
        rows = read_json(out / "posterior.json")["rows"]
        return int(max(rows, key=lambda r: r["alpha__posterior_mean"])["layer"])
    return default


def bayes_cache(path: Path, fams: dict, datasets: dict, checkpoints, edges, K, device) -> dict:
    """Bayes posterior predictives for every (prior family, data family) pair, cached on disk."""
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["bayes"].item()
    out = {}
    for pf, pd in fams.items():
        for df, b in datasets.items():
            print(f"[bayes] prior {pf} on {df} data ({len(b['times64'])} sequences)", flush=True)
            out[(pf, df)] = r7.bayes_dists(pd, b["times64"], checkpoints, edges, K, 4242, device)
    np.savez(path, bayes=np.array(out, dtype=object))
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
    if args.smoke:
        main_cfg = deep_update(main_cfg, {"run_name": main_cfg["run_name"] + "_r7smoke", "data": {"n_events": 64},
                                          "model": {"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}})
        keep, seen = [], set()
        for r in main_cfg["runs"]:
            k = task.kernel_of(deep_update(main_cfg["data"], r.get("data", {})))
            if k not in seen:
                seen.add(k)
                keep.append(r)
        main_cfg["runs"] = keep
        cfg = deep_update(cfg, {"misspec": {"n_seqs": 6, "n_fit": 8, "K": 300, "checkpoints": [8, 32]},
                                "interchange": {"n_fit": 8, "n_pairs": 6, "K": 300, "checkpoints": [16, 32], "layers": [1]},
                                "bank": {"n_seqs": 12, "n_test": 6, "K": 300}})
        smoke_steps = 20
    out_root = ensure_dir(Path(main_cfg.get("out_root", "outputs")) / main_cfg["run_name"])
    rep_dir = ensure_dir(Path(main_cfg.get("out_root", "outputs")) / (cfg.get("run_name", "hawkes_r7") + ("_smoke" if args.smoke else "")))
    device = get_device("auto")
    print(f"mechlab {__version__} | torch {torch.__version__} | device {device}")
    edges = bin_edges(int(main_cfg["data"].get("n_bins", 48)))
    runs = [(n, d, m) for n, d, m in run_cfgs(main_cfg) if not args.only or n in args.only]
    fams = {}
    for n, d, m in run_cfgs(main_cfg):
        fams.setdefault(task.kernel_of(d), {k: v for k, v in d.items() if k != "batch_size"})
    mc, ic, kc = cfg.get("misspec", {}), cfg.get("interchange", {}), cfg.get("bank", {})

    if "misspec" in args.stages:
        cps = [int(x) for x in mc.get("checkpoints", [8, 16, 32, 64, 128])]
        cps = [m for m in cps if m <= int(main_cfg["data"]["n_events"]) - 1]
        datasets = {f: task.make_batch({**d, "batch_size": int(mc.get("n_seqs", 200))}, [77, 7 if f == "exp" else 8], edges,
                                       with_truth=True) for f, d in fams.items()}
        bayes = bayes_cache(rep_dir / "bayes_cache.npz", fams, datasets, cps, edges, int(mc.get("K", 20000)), device) \
            if len(fams) == 2 else None
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if bayes is None or (out / "r7_misspec.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            layer = best_posterior_layer(out, min(3, mcfg["n_layers"]))
            fit = r7.fit_alpha_probe(model, dcfg, edges, int(mc.get("n_fit", 300)), 11, device, layer,
                                     int(mc.get("K", 20000)), cps)
            res = r7.misspecification(model, dcfg, datasets, bayes, edges, cps, device, layer, fit)
            write_json(out / "r7_misspec.json", res)
            for fam, r in res["data"].items():
                print(f"[{name}] misspec on {fam} data: KL to Bayes-own {r['kl_own']:.4f} | to Bayes-true {r['kl_true']:.4f} | "
                      f"alpha corr decoded~own {r['alpha_corr_decoded_own']:.3f} decoded~true {r['alpha_corr_decoded_true']:.3f}",
                      flush=True)
    if "interchange" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if task.kernel_of(dcfg) != "exp" or (out / "r7_interchange.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            layers = [l for l in ic.get("layers", [1, 2, 3]) if l < mcfg["n_layers"] - 1 or args.smoke]
            res = r7.interchange(model, dcfg, edges, int(ic.get("n_fit", 400)), int(ic.get("n_pairs", 300)), 21, device,
                                 layers=layers, checkpoints=tuple(ic.get("checkpoints", [64, 128])), K=int(ic.get("K", 20000)))
            write_json(out / "r7_interchange.json", res)
            for li, r in res["layers"].items():
                print(f"[{name}] interchange after block {li}: transfer {r['transfer_posterior']:.3f} (random subspace "
                      f"{r['transfer_random']:.3f})", flush=True)
    if "bank" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if (out / "r7_bank.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            res = r7.kernel_bank(model, dcfg, edges, int(kc.get("n_seqs", 400)), 31, device, K=int(kc.get("K", 20000)),
                                 n_test=int(kc.get("n_test", 400)))
            write_json(out / "r7_bank.json", res)
            for k, v in res["sufficiency"].items():
                print(f"[{name}] bank {k}: bank MLP R2 {v['bank_mlp']:.3f} | count+time MLP {v['count_time_mlp']:.3f} | "
                      f"best residual probe {v['residual_best_linear']:.3f}", flush=True)
            for r in res["reconstruction"]:
                print(f"[{name}] bank reconstruction after block {r['block']}: CE {r['ce_original']:.4f} -> bank {r['ce_bank']:.4f}, "
                      f"count+time {r['ce_count_time']:.4f}, shuffled bank {r['ce_bank_shuffled']:.4f}", flush=True)
    if "report" in args.stages:
        print(report(cfg, main_cfg, out_root, rep_dir))


def report(cfg, main_cfg, out_root, rep_dir) -> str:
    th = cfg.get("verdict", {})
    lines, verdicts = [f"# Hawkes round 7: misspecification, posterior interchange, kernel bank", "", f"mechlab {__version__}", ""], []
    rows, mis = [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r7_misspec.json"
        if not p.exists():
            continue
        r = read_json(p)
        for fam, d in r["data"].items():
            rows.append([name, r["own_family"], fam, "yes" if d["misspecified"] else "no", f"{d['kl_own']:.4f}", f"{d['kl_true']:.4f}",
                         f"{d['ce_model']:.3f}", f"{d['ce_bayes_own']:.3f}", f"{d['ce_bayes_true']:.3f}",
                         f"{d['alpha_corr_decoded_own']:.3f}", f"{d['alpha_corr_decoded_true']:.3f}"])
            if d["misspecified"]:
                mis.append((name, d))
    if rows:
        lines += ["## M: misspecification", "", "KL(Bayes || model) of the next-interval forecast, averaged over context "
                  "lengths; Bayes-own uses the model's training prior, Bayes-true the prior of the family that generated "
                  "the data. alpha corr: decoded branching ratio vs the own-family posterior mean and vs the true value.", "",
                  md_table(["run", "trained on", "data", "misspecified", "KL own", "KL true", "CE model", "CE Bayes own",
                            "CE Bayes true", "alpha corr (own post.)", "alpha corr (true)"], rows), ""]
        m1 = [d["kl_own"] < th.get("m1_ratio", 0.5) * d["kl_true"] for _, d in mis]
        m2 = [d["alpha_corr_decoded_own"] > d["alpha_corr_decoded_true"] for _, d in mis]
        if mis:
            verdicts.append(verdict_line("M1 the model forecasts like the Bayes predictive of its own prior", sum(m1) >= 0.75 * len(m1),
                                         f"{sum(m1)}/{len(m1)} models with KL-own < {th.get('m1_ratio', 0.5)} x KL-true on misspecified data"))
            verdicts.append(verdict_line("M2 its decoded branching ratio tracks its own posterior", sum(m2) >= 0.75 * len(m2),
                                         f"{sum(m2)}/{len(m2)} models with corr(decoded, own posterior) > corr(decoded, true)"))
    rows, ok = [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r7_interchange.json"
        if not p.exists():
            continue
        r = read_json(p)
        for li, v in r["layers"].items():
            rows.append([name, li, f"{v['transfer_posterior']:.3f}", f"{v['transfer_random']:.3f}",
                         " / ".join(f"{v[f'posterior_m{m}']['kl_transplant_original']:.3f} -> {v[f'posterior_m{m}']['kl_transplant_patched']:.3f}"
                                    for m in r["checkpoints"])])
        b = r["layers"][str(r["best_layer"])] if str(r["best_layer"]) in r["layers"] else r["layers"][r["best_layer"]]
        ok.append(b["transfer_posterior"] >= th.get("i1_transfer", 0.5)
                  and b["transfer_posterior"] >= th.get("i1_ratio", 3.0) * max(b["transfer_random"], 1e-6))
    if rows:
        lines += ["## I: posterior interchange", "", "Transfer = 1 - KL(Bayes-transplant || patched) / KL(Bayes-transplant || "
                  "original): 1 means the patched forecast equals the Bayes forecast with A's posterior on B's history.", "",
                  md_table(["run", "after block", "transfer (posterior subspace)", "transfer (random subspace)",
                            "KL to Bayes-transplant before -> after, per context"], rows), ""]
        verdicts.append(verdict_line("I1 swapping the posterior subspace transplants the posterior", sum(ok) >= 0.75 * len(ok),
                                     f"{sum(ok)}/{len(ok)} models (transfer >= {th.get('i1_transfer', 0.5)} and >= "
                                     f"{th.get('i1_ratio', 3.0)}x random)"))
    rows, k1, k2 = [], [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r7_bank.json"
        if not p.exists():
            continue
        r = read_json(p)
        s = r["sufficiency"]
        valid = [x for x in r["reconstruction"] if x.get("recovered") is not None]
        rec = max(valid, key=lambda x: x["recovered"]) if valid else None
        rows.append([name] + [f"{s[k]['bank_mlp']:.3f} / {s[k]['count_time_mlp']:.3f} / {s[k]['residual_best_linear']:.3f}"
                              for k in r7.PARAMS]
                    + ([f"{rec['block']}", f"{rec['ce_original']:.4f}", f"{rec['ce_bank']:.4f}", f"{rec['ce_count_time']:.4f}",
                        f"{rec['ce_bank_shuffled']:.4f}", f"{rec['recovered']:.2f}"] if rec else ["-"] * 6))
        k1.append(all(s[k]["bank_mlp"] >= s[k]["residual_best_linear"] - th.get("k1_tol", 0.03) for k in r7.PARAMS))
        if rec:
            k2.append(rec["recovered"] >= th.get("k2_recovered", 0.7))
    if rows:
        lines += ["## K: kernel bank", "", "Posterior-mean R2 from features: bank (decayed counts at 7 rates + count, time) "
                  "MLP / count+time MLP / best linear probe on the residual stream. Reconstruction: replace the output of one "
                  "block by a linear map of the features and measure late-context cross-entropy (nats); recovered = share of the "
                  "count+time damage that the bank removes (best block; blocks where count+time costs < 0.01 nats are skipped).", "",
                  md_table(["run", "log mu", "alpha", "log beta", "block", "CE original", "CE bank", "CE count+time",
                            "CE shuffled bank", "recovered"], rows), ""]
        verdicts.append(verdict_line("K1 the bank suffices for the posterior", sum(k1) >= 0.75 * len(k1), f"{sum(k1)}/{len(k1)} models"))
        if k2:
            verdicts.append(verdict_line("K2 a block's output is a linear function of the bank", sum(k2) >= 0.75 * len(k2),
                                         f"{sum(k2)}/{len(k2)} models recover >= {th.get('k2_recovered', 0.7)} of the damage"))
    text = "\n".join(lines[:4] + ["## Verdict", ""] + (verdicts or ["(no results yet)"]) + [""] + lines[4:])
    (rep_dir / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    main()
