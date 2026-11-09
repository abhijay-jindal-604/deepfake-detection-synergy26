#!/usr/bin/env bash
# Overnight GPU chain (sequential): PHASE3 step 2 5-fold queue (cnxs -> b4_320 -> vitb), PHASE4 B0 ablations
# (3-fold, queues/p4_b0.yaml), LOCO for p2_final (queues/loco_p2_final.yaml), then flip-TTA OOF + val-C evals
# (PHASE5 §4.3) for every finished fold of the P3 runs and p2_final. Restartable: run_queue skips finished
# folds, TTA evals skip folds whose tta_flip/eval.json exists.
#   DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset nohup scripts/p3_step2.sh > outputs/runs/p3_step2.stdout 2>&1 &
set -u
cd "$(dirname "$0")/.."
: "${DF_DATA_ROOT:?set DF_DATA_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD/src"
PY=.venv/bin/python
echo "$(date -Is) chain start"
for q in p3_final p4_b0 loco_p2_final; do
  echo "$(date -Is) queue $q start"
  $PY scripts/run_queue.py queues/$q.yaml
  rc=$?; echo "$(date -Is) queue $q exit $rc"
  [ $rc -eq 130 ] && { echo "interrupted"; exit 130; }
done
for run in cnxs_final b4_320_final vitb_final p2_final; do
  for k in 0 1 2 3 4; do
    d=outputs/runs/$run/fold$k
    [ -f "$d/last.pt" ] && [ -f "$d/eval.json" ] || { echo "$(date -Is) TTA skip $run/fold$k (not trained)"; continue; }
    [ -f "$d/tta_flip/eval.json" ] && { echo "$(date -Is) TTA done $run/fold$k"; continue; }
    echo "$(date -Is) TTA start $run/fold$k"
    timeout 30m $PY -m dfd.evaluate --ckpt "$d/last.pt" --tta_flip > "$d/tta_flip.log" 2>&1
    echo "$(date -Is) TTA end $run/fold$k exit $?"
  done
done
echo "$(date -Is) chain done"
