# Transformers Pool the Likelihood

Code and results for *Transformers Pool the Likelihood: Inside a Bayes-Optimal In-Context Forecaster of
Self-Exciting Processes* (Ashish Goel, 2026). The arXiv link will be added here when the preprint is posted.

Small transformers (0.8M parameters) are trained to forecast the next event time of Hawkes processes whose parameters
are drawn afresh for every sequence, so the parameters must be inferred in context. The models match the exact
Bayes-optimal forecaster at every context length. On data from another kernel family they forecast like the Bayes
predictive of their *own* training prior. They carry the posterior in a three-dimensional subspace that the forecast
reads causally, and they compute it by averaging log-transformed decayed event counts over past events, the sum that
appears in the Hawkes log-likelihood.

## Contents

| Path | What it holds |
|---|---|
| `mechlab/core/` | Transformer with time-rotary attention and an optional null attention slot; resumable training; grouped ridge and MLP probes; report helpers |
| `mechlab/hawkes/` | Exact simulation (exponential kernel: Dassios–Zhao; power law: Ogata thinning), next-interval targets, Bayes-optimal and maximum-likelihood forecasters, posterior probes, parameter edits, misspecification, interchange and distributed alignment search (DAS), pooled traces, head ablation, FIM-PP comparison |
| `configs/` | One config per experiment round; each lists its predictions and thresholds in its comments |
| `scripts/` | Entry points for each round, plus `smoke_test.sh` |
| `tests/` | Unit tests and tiny end-to-end runs |
| `paper/data/` | Every result file behind a number, table or figure in the paper |
| `paper/scripts/` | Regenerate the paper's numbers, tables and figures from `paper/data/` |
| `docs/PREREGISTRATION.md` | The predictions fixed before each round, and their outcomes, including the failures |

## Install

Python 3.10 or later. Install PyTorch for your platform first (https://pytorch.org), then:

```
pip install -r requirements.txt
```

## Quick check (CPU, a few minutes)

```
python -m pytest -q tests
bash scripts/smoke_test.sh        # tiny models through every stage of every round
```

## Regenerate the paper's numbers

```
python paper/scripts/make_numbers.py    # writes paper/numbers.tex and paper/tables/*.tex from paper/data/
python paper/scripts/make_figures.py    # writes paper/figures/*.pdf (needs matplotlib)
```

The committed `paper/numbers.tex` and `paper/tables/` are the versions used in the paper. After regenerating,
`git diff` should show no changes.

## Rerun the experiments

The reported runs used about 16 GPU-hours on free NVIDIA T4s (Kaggle). Every stage writes its results as JSON and is
skipped on a rerun, so a command that stops (for example at Kaggle's session limit) can be started again.

```
python scripts/hawkes_run.py --config configs/hawkes_main.yaml   # train the 8 models, then evaluation, probes,
                                                                 # kernel fits, mechanism, edits, Bayes baselines,
                                                                 # posterior probes and a report
python scripts/hawkes_r7.py  --config configs/hawkes_r7.yaml     # misspecification, interchange, decayed counts
python scripts/hawkes_r8.py  --config configs/hawkes_r8.yaml     # pooled traces, DAS, misspecification intervals
python scripts/hawkes_r9.py  --config configs/hawkes_r9.yaml     # head ablation, DAS controls
python scripts/fimpp_run.py  --config configs/hawkes_r7.yaml     # optional FIM-PP comparison (see below)
python paper/scripts/collect_results.py                          # copy outputs/ into paper/data/
```

- **Two GPUs:** run two processes, each with its own `CUDA_VISIBLE_DEVICES` and a disjoint `--only` list of run names
  (for example `--only null_s0 null_s1 nonull_s0 notime_s0` and `--only null_s2 powerlaw_s0 powerlaw_s1 powerlaw_s2`).
- **Kaggle:** use a notebook with GPU T4 x2 and Internet on. Run `!git clone` on this repository, `%cd` into it, and run
  the commands above with `!python ...`.
- **FIM-PP comparison:** this needs internet access (it downloads `FIM4Science/fim-pp`) and the OpenFIM source code.
  Point the `FIM_SRC` environment variable at it, or place it at `./OpenFIM/src`.
- **Reproducibility:** results are not bit-identical across GPU types and library versions. Expect agreement within
  the spread across seeds reported in the paper.

## Citation

```
@misc{goel2026pool,
  title  = {Transformers Pool the Likelihood: Inside a {Bayes}-Optimal In-Context Forecaster of Self-Exciting Processes},
  author = {Goel, Ashish},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT (see `LICENSE`).
