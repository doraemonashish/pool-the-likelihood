#!/usr/bin/env bash
# Offline check: unit tests plus tiny models through every stage of every round (CPU, no internet, a few minutes).
set -e
cd "$(dirname "$0")/.."
python -m pytest -q tests 2>&1 | tail -1
python scripts/hawkes_run.py --config configs/hawkes_main.yaml --smoke > /tmp/smoke_hawkes.log 2>&1  && echo "main pipeline: OK" || { tail -20 /tmp/smoke_hawkes.log; exit 1; }
python scripts/hawkes_r7.py  --config configs/hawkes_r7.yaml   --smoke > /tmp/smoke_hawkes7.log 2>&1 && echo "round 7:       OK" || { tail -20 /tmp/smoke_hawkes7.log; exit 1; }
python scripts/hawkes_r8.py  --config configs/hawkes_r8.yaml   --smoke > /tmp/smoke_hawkes8.log 2>&1 && echo "round 8:       OK" || { tail -20 /tmp/smoke_hawkes8.log; exit 1; }
python scripts/hawkes_r9.py  --config configs/hawkes_r9.yaml   --smoke > /tmp/smoke_hawkes9.log 2>&1 && echo "round 9:       OK" || { tail -20 /tmp/smoke_hawkes9.log; exit 1; }
rm -rf outputs/*_smoke outputs/*smoke
echo "All Hawkes pipelines run end to end."
