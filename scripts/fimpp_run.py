#!/usr/bin/env python
"""Exploratory audit of the FIM-PP point-process foundation model against Bayes (see mechlab/hawkes/fimpp.py).

    python scripts/fimpp_run.py --config configs/hawkes_r7.yaml                 # downloads FIM-PP (internet)
    python scripts/fimpp_run.py --config configs/hawkes_r7.yaml --random-model OpenFIM/configs/train/hawkes/david.yaml --smoke

Needs the OpenFIM source on PYTHONPATH (FIM_SRC environment variable or ./OpenFIM/src).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for cand in (os.environ.get("FIM_SRC"), "OpenFIM/src", "../OpenFIM/src", "/kaggle/working/OpenFIM/src"):
    if cand and Path(cand).exists():
        sys.path.insert(0, str(Path(cand).resolve()))
        break

import torch  # noqa: E402

from mechlab import __version__  # noqa: E402
from mechlab.core.report import md_table  # noqa: E402
from mechlab.core.utils import deep_update, ensure_dir, get_device, load_config, write_json  # noqa: E402
from mechlab.hawkes import fimpp, task  # noqa: E402
from mechlab.hawkes.simulate import bin_edges  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--random-model", default=None, help="OpenFIM training yaml: random weights (offline test)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    main_cfg = load_config(cfg["main_config"])
    fc = deep_update({"Ks": [1, 4, 16, 64, 256], "n_proc": 200, "n_events": 100, "n_prior": 20000, "max_k_bayes": 16,
                      "checkpoints": [2, 4, 8, 16, 32, 64, 96]}, cfg.get("fimpp", {}))
    if args.smoke:
        fc.update({"Ks": [1, 4], "n_proc": 4, "n_events": 40, "n_prior": 300, "checkpoints": [2, 8, 32]})
    out = ensure_dir(Path(main_cfg.get("out_root", "outputs")) / ("fimpp" + ("_smoke" if args.smoke else "")))
    device = get_device("auto")
    print(f"mechlab {__version__} | torch {torch.__version__} | device {device}")
    if args.random_model:
        model, source = fimpp.random_model(args.random_model, device), "random weights (test)"
    else:
        model, source = fimpp.load_pretrained(device)
    print(f"[fimpp] model: {source}, {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    edges = bin_edges(int(main_cfg["data"].get("n_bins", 48)))
    own = None
    own_path = Path(main_cfg.get("out_root", "outputs")) / main_cfg["run_name"] / "null_s0" / "model.pt"
    if own_path.exists():
        own = task.build_model(main_cfg["model"], len(edges) - 1).to(device)
        own.load_state_dict(torch.load(own_path, map_location=device))
        own.eval()
    res = fimpp.audit(model, edges, device, main_cfg["data"].get("params", {}), Ks=fc["Ks"], n_proc=int(fc["n_proc"]),
                      n_events=int(fc["n_events"]), checkpoints=fc["checkpoints"], n_prior=int(fc["n_prior"]),
                      max_k_bayes=int(fc["max_k_bayes"]), own_model=own)
    res["source"] = source
    write_json(out / "fimpp.json", res)
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    rows = [[r["K"], f(r["ce_fimpp"]), f(r.get("ce_bayes")), f(r["ce_oracle"]), f(r.get("kl_bayes_fimpp")),
             f(r.get("log_beta_corr_est_posterior")), f(r.get("log_beta_corr_est_true")),
             f(r.get("log_mu_corr_est_posterior")), f(r.get("log_mu_corr_est_true")),
             f(r.get("log_beta_probe_r2_posterior")), f(r.get("log_beta_r2_posterior_vs_true"))] for r in res["rows"]]
    text = "\n".join([f"# FIM-PP audit (exploratory)", "", f"Model: {source}. Data: our exponential-kernel prior; K context "
                      f"paths of {fc['n_events']} events plus one inference path. Cross-entropy (nats) of the next interval, "
                      "averaged over context lengths.", "",
                      md_table(["K", "CE FIM-PP", "CE Bayes", "CE oracle", "KL(Bayes to FIM-PP)", "corr(log beta est, posterior)",
                                "corr(log beta est, true)", "corr(log mu est, posterior)", "corr(log mu est, true)",
                                "probe R2 log beta (posterior)", "ceiling log beta"], rows), "",
                      (f"Own transformer with no context paths: CE {f(res['rows'][0].get('ce_own_transformer_no_context'))} vs "
                       f"Bayes with no context {f(res['rows'][0].get('ce_bayes_no_context'))}.") if own is not None else ""])
    (out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
