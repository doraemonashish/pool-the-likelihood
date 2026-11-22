#!/usr/bin/env python
"""Hawkes round 9: pooling-head ablation (A1) and DAS controls (D1-D3) on the exponential-kernel models.
Needs outputs/hawkes_main/<run>/model.pt and r8_pooled.json (for the pooling heads).

    python scripts/hawkes_r9.py --config configs/hawkes_r9.yaml
    python scripts/hawkes_r9.py --config configs/hawkes_r9.yaml --smoke
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
from mechlab.hawkes import r8, r9  # noqa: E402
from mechlab.hawkes import task  # noqa: E402
from mechlab.hawkes.simulate import bin_edges  # noqa: E402

STAGES = ["ablate", "das", "report"]


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
        main_cfg = deep_update(main_cfg, {"run_name": main_cfg["run_name"] + "_r9smoke", "data": {"n_events": 64},
                                          "model": {"d_model": 32, "n_layers": 3, "n_heads": 2, "d_mlp": 64}})
        main_cfg["runs"] = main_cfg["runs"][:1]
        cfg = deep_update(cfg, {"ablate": {"n_seqs": 8, "K": 300, "n_random": 2, "query_from": 24},
                                "das": {"n_fit": 8, "n_pairs": 8, "K": 300, "checkpoints": [16, 32], "steps": 2, "blocks": [0, 1]}})
        smoke_steps = 20
    base = Path(main_cfg.get("out_root", "outputs"))
    out_root = ensure_dir(base / main_cfg["run_name"])
    rep_dir = ensure_dir(base / (cfg.get("run_name", "hawkes_r9") + ("_smoke" if args.smoke else "")))
    device = get_device("auto")
    print(f"mechlab {__version__} | torch {torch.__version__} | device {device}")
    edges = bin_edges(int(main_cfg["data"].get("n_bins", 48)))
    runs = [(n, d, m) for n, d, m in run_cfgs(main_cfg) if (not args.only or n in args.only) and task.kernel_of(d) == "exp"]
    ac, dc = cfg.get("ablate", {}), cfg.get("das", {})
    tv = float(cfg.get("verdict", {}).get("pool_tv", 0.25))
    if "ablate" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if (out / "r9_ablate.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            if (out / "r8_pooled.json").exists():
                heads = r9.pooling_heads(read_json(out / "r8_pooled.json")["attention"], tv)
            else:                                                   # smoke / missing: find them now
                col = task.collect(model, dcfg, edges, 16, 31, device, query_from=int(ac.get("query_from", 48)))
                has_null = col["attn"][0].shape[-1] == col["attn"][0].shape[-2] + 1
                rows = r8.attention_pooling(col["attn"], col["query_idx"], has_null)
                heads = r9.pooling_heads(rows, tv) or [(min(1, len(model.blocks) - 1), 0)]
            res = r9.head_ablation(model, dcfg, edges, heads, int(ac.get("n_seqs", 400)), 41, device, K=int(ac.get("K", 20000)),
                                   n_random=int(ac.get("n_random", 10)), query_from=int(ac.get("query_from", 48)))
            write_json(out / "r9_ablate.json", res)
            s = res.get("summary", {})
            print(f"[{name}] pooling heads {res['heads']}: CE +{s.get('dce_pooling', 0):.4f} (random heads mean "
                  f"+{s.get('dce_random_mean', 0):.4f}, max +{s.get('dce_random_max', 0):.4f}) | pooled-trace R2 drop "
                  f"{s.get('pooled_r2_drop_pooling', 0):.3f} vs {s.get('pooled_r2_drop_random_mean', 0):.3f}", flush=True)
    if "das" in args.stages:
        for name, dcfg, mcfg in runs:
            out = out_root / name
            if (out / "r9_das_controls.json").exists():
                continue
            model = load_model(out, dcfg, mcfg, edges, device, smoke_steps)
            if model is None:
                continue
            res = r9.das_controls(model, dcfg, edges, int(dc.get("n_fit", 400)), int(dc.get("n_pairs", 600)), 21, device,
                                  blocks=tuple(dc.get("blocks", [0, 1, 2])), k=int(dc.get("k", 3)),
                                  checkpoints=tuple(dc.get("checkpoints", [64, 128])), K=int(dc.get("K", 20000)),
                                  steps=int(dc.get("steps", 300)))
            write_json(out / "r9_das_controls.json", res)
            for r in res["blocks"]:
                print(f"[{name}] block {r['block']}: right target {r['right_target_transfer']:.3f} | wrong target "
                      f"{r['wrong_target_transfer']:.3f} | random {r['random_subspace_right_target']:.3f} | posterior R2 from 3 DAS "
                      f"coords {' '.join(f'{x:.2f}' for x in r['posterior_r2_from_das_coords'])}", flush=True)
    if "report" in args.stages:
        print(report(cfg, main_cfg, out_root, rep_dir))


def report(cfg, main_cfg, out_root, rep_dir) -> str:
    th = cfg.get("verdict", {})
    lines, v = ["# Hawkes round 9: pooling-head ablation and DAS controls", "", f"mechlab {__version__}", ""], []
    rows, a1, a2 = [], [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r9_ablate.json"
        if not p.exists():
            continue
        r = read_json(p)
        if not r.get("pooling"):
            continue
        s = r["summary"]
        rows.append([name, " ".join(f"b{b}h{h}" for b, h in r["heads"]), f"{r['baseline']['ce']:.4f}", f"+{s['dce_pooling']:.4f}",
                     f"+{s['dce_random_mean']:.4f} (max +{s['dce_random_max']:.4f})",
                     f"{r['baseline']['pooled_r2']:.3f} -> {r['pooling']['pooled_r2']:.3f}",
                     f"{np.mean([x['pooled_r2'] for x in r['random']]):.3f}",
                     " ".join(f"{x:.2f}" for x in r["pooling"]["posterior_r2_final"])])
        a1.append(s["dce_pooling"] >= th.get("a1_ratio", 3.0) * max(s["dce_random_mean"], 1e-4) and s["dce_pooling"] > s["dce_random_max"])
        a2.append(s["pooled_r2_drop_pooling"] - s["pooled_r2_drop_random_mean"] >= th.get("a2_gap", 0.05))
    if rows:
        lines += ["## A: mean-ablating the pooling heads", "", "Late-context cross-entropy (nats) and pooled-trace R2 at the output "
                  "of the ablated heads' block (probes refit on the ablated model), against ablating the same number of other heads "
                  "in the same blocks (10 draws). Last column: posterior-mean R2 at the final layer after ablating the pooling heads.", "",
                  md_table(["run", "pooling heads", "CE", "CE change (pooling)", "CE change (other heads)", "pooled R2 (pooling)",
                            "pooled R2 (other heads)", "posterior R2 after"], rows), ""]
        v.append(verdict_line("A1 the pooling heads are necessary", sum(a1) >= 0.75 * len(a1),
                              f"{sum(a1)}/{len(a1)} models: CE rise >= {th.get('a1_ratio', 3.0)}x the other-heads mean and above their max"))
        v.append(verdict_line("A2 they carry the pooled traces", sum(a2) >= 0.75 * len(a2),
                              f"{sum(a2)}/{len(a2)} models: pooled-trace R2 drops >= {th.get('a2_gap', 0.05)} more than for other heads"))
    rows, d1, d2, al = [], [], [], []
    for name, dcfg, _ in run_cfgs(main_cfg):
        p = out_root / name / "r9_das_controls.json"
        if not p.exists():
            continue
        r = read_json(p)
        for b in r["blocks"]:
            rows.append([name, b["block"], f"{b['right_target_transfer']:.3f}", f"{b['wrong_target_transfer']:.3f}",
                         f"{b['right_subspace_on_wrong_target']:.3f}", f"{b['random_subspace_right_target']:.3f}",
                         " ".join(f"{x:.2f}" for x in b["posterior_r2_from_das_coords"]),
                         " ".join(f"{x:.2f}" for x in b["posterior_r2_from_random_coords"]),
                         " ".join(f"{x:.2f}" for x in b["posterior_r2_full_residual"])])
        best = max(r["blocks"], key=lambda b: b["right_target_transfer"])
        d1.append(best["wrong_target_transfer"] <= th.get("d1_wrong", 0.3) and best["right_target_transfer"] >= th.get("d1_right", 0.8))
        d2.append(min(best["posterior_r2_from_das_coords"]) >= th.get("d2_r2", 0.8))
        al += [x["overlap"] for x in r["alignment"]]
        rows_al = r["alignment_random_expected"]
    if rows:
        lines += ["## D: DAS controls (3-dim subspaces, prefix patching)", "", "Transfer on held-out pairs. Wrong target: the subspace is "
                  "trained to give the Bayes forecast with an unrelated sequence C's posterior while patching from A. Posterior R2 "
                  "(log mu, alpha, log beta) from the 3 DAS coordinates, from 3 random coordinates, and from the whole residual.", "",
                  md_table(["run", "block", "right target", "wrong target", "right subspace, wrong target", "random subspace",
                            "posterior R2 (DAS 3)", "posterior R2 (random 3)", "posterior R2 (all)"], rows), "",
                  f"Alignment of the DAS subspaces across blocks (share of one inside the other): mean {np.mean(al):.2f} "
                  f"(random {rows_al:.2f}).", ""]
        v.append(verdict_line("D1 DAS cannot produce an unrelated posterior", sum(d1) >= 0.75 * len(d1),
                              f"{sum(d1)}/{len(d1)} models: wrong-target transfer <= {th.get('d1_wrong', 0.3)} where the right target "
                              f"reaches >= {th.get('d1_right', 0.8)}"))
        v.append(verdict_line("D2 the DAS coordinates carry the posterior", sum(d2) >= 0.75 * len(d2),
                              f"{sum(d2)}/{len(d2)} models: R2 >= {th.get('d2_r2', 0.8)} for all three posterior means"))
        v.append(f"- **D3 alignment across blocks (exploratory)**: mean overlap {np.mean(al):.2f} vs {rows_al:.2f} at random")
    text = "\n".join(lines[:4] + ["## Verdict", ""] + (v or ["(no results yet)"]) + [""] + lines[4:])
    (rep_dir / "report.md").write_text(text)
    return text


if __name__ == "__main__":
    main()
