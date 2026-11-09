#!/usr/bin/env bash
# Full-data retrains (PHASE5 §3, --fold full, fixed epochs from each *_final config) of the Phase 3 backbones,
# sequential (one GPU job at a time). p2_final/full already exists. Skips runs whose full/done.json exists.
#   DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset nohup setsid scripts/p5_full.sh > outputs/runs/p5_full.stdout 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
: "${DF_DATA_ROOT:?set DF_DATA_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD/src"
PY=.venv/bin/python
echo "$(date -Is) chain start"
for run in cnxs_final b4_320_final vitb_final; do
  [ -f "outputs/runs/$run/full/done.json" ] && { echo "$(date -Is) skip $run (done)"; continue; }
  echo "$(date -Is) $run full start"
  $PY -m dfd.train --config configs/p3/$run.yaml --fold full --run_name $run > outputs/runs/${run}_full.log 2>&1
  echo "$(date -Is) $run full exit $?"
done
echo "$(date -Is) chain done"
