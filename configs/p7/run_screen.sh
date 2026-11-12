#!/usr/bin/env bash
# P7 screening chain: B0 p2_final + {mixup, res288} on folds 0,1,2, sequential (one GPU job at a time).
# A failure of one run does not stop the next; --skip_done skips finished folds.
#   nohup setsid bash configs/p7/run_screen.sh > outputs/runs/p7_screen.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
export DF_DATA_ROOT="${DF_DATA_ROOT:-resources/extracted/dataset/hackathon_dataset}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD/src"
PY=.venv/bin/python
echo "$(date -Is) chain start (pid $$)"
for run in p7_mixup p7_res288; do
  echo "$(date -Is) $run folds 0,1,2 start"
  $PY -m dfd.train --config configs/p7/$run.yaml --fold 0,1,2 --run_name $run --skip_done > outputs/runs/$run.log 2>&1
  echo "$(date -Is) $run exit $?"
done
echo "$(date -Is) chain done"
