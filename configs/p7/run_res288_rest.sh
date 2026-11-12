#!/usr/bin/env bash
# P7 res288 completion: folds 3,4 then the full-data retrain (mirrors scripts/p5_full.sh: --fold full --run_name <run>),
# sequential (one GPU job at a time). A fold failure does not block the full retrain; --skip_done skips finished work.
# Output: outputs/runs/p7_res288/fold{3,4}, outputs/runs/p7_res288/full/{last.pt,done.json}
#   nohup setsid bash configs/p7/run_res288_rest.sh > outputs/runs/p7_res288_rest.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
export DF_DATA_ROOT="${DF_DATA_ROOT:-resources/extracted/dataset/hackathon_dataset}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD/src"
PY=.venv/bin/python
run=p7_res288
echo "$(date -Is) chain start (pid $$)"
echo "$(date -Is) $run folds 3,4 start"
$PY -m dfd.train --config configs/p7/$run.yaml --fold 3,4 --run_name $run --skip_done > outputs/runs/${run}_f34.log 2>&1
echo "$(date -Is) $run folds 3,4 exit $?"
echo "$(date -Is) $run full start"
$PY -m dfd.train --config configs/p7/$run.yaml --fold full --run_name $run --skip_done > outputs/runs/${run}_full.log 2>&1
echo "$(date -Is) $run full exit $?"
echo "$(date -Is) chain done"
