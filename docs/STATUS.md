# Coordinator status

Last update: 2026-11-09 13:45 IST (GPU build agent).

## Phase / step
Phase 2 COMPLETE (p2_final 5-fold OOF AUC 0.8964; recipe `configs/p2/p2_final.yaml`). Phase 5 code merged to main (f61c07a); CPU + GPU FP16 parity verified. **Safety-net submission v0_b0 exists** (`outputs/submissions/v0_b0/`, single B0 full-data retrain, validator ALL PASS). Phase 3 step 0 done; **step 1 (LR sanity, fold 0) running**; step 2 (5-fold cnxs → b4_320 → vitb) follows.

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
- GPU queue `queues/p3_lr.yaml` (6 fold-0 runs: cnxs_lr1e4, cnxs_lr5e5, b4_320_lr2e4, b4_320_lr1e4, vitb_lr5e5, vitb_lr1e4), started 13:36. Log `outputs/runs/queue_p3_lr.log`, per-run `outputs/runs/<run>.log`. Resume (skips finished folds): `DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset nohup .venv/bin/python scripts/run_queue.py queues/p3_lr.yaml > outputs/runs/queue_p3_lr.stdout 2>&1 &`
- Then: LR pick per PHASE3 rule (fold-0 selection; within 0.01 → lower LR) → `configs/p3/*_final.yaml` → `queues/p3_final.yaml`.

## Safety net (v0_b0)
`outputs/submissions/v0_b0/predictions.csv` + `model_weights.pt` (15.6 MiB) + run logs + `validate.txt`. Rebuild: `python scripts/bundle.py --member p2_final:full --out <dir>/model_weights.pt`, then `python inference_script.py --test_dir <test> --weights <dir>/model_weights.pt --output <dir>/predictions.csv`. Local GPU 13.8 s wall, byte-identical re-run. Not yet produced on a T4 (Kaggle kit: `scripts/kaggle/`).

## Notes
- `run_queue.py` does not handle `--fold full`; launch full-data retrains with `nohup .venv/bin/python -m dfd.train --config <cfg> --fold full --run_name <run>` directly.
- Inference throughput (B0, 3k imgs): ~5 s fixed CUDA/worker warm-up + ~550 img/s; data loading (4 workers) is the bottleneck for B0, not the GPU.
- Don't redirect run_queue stdout to the queue's own log file (it writes `outputs/runs/queue_<name>.log` itself; lines get duplicated).

## Pending user decisions
- Approve audit grids: `outputs/audit/pair_masks.jpg` and `outputs/audit/sbi_examples.jpg`. `data/masks*`, `data/mask_prior.npy` stay uncommitted until approved (needed for p4 configs).

## Next actions
1. After step 1: commit LR decisions, launch `queues/p3_final.yaml` (cnxs → b4_320 → vitb, 5 folds) + flip-TTA OOF evals.
2. Phase 3 summary (compare.py, OOF correlations) → ensemble pool; LOCO queue (`queues/loco_p2_final.yaml`, coordinator schedules after Phase 3).
3. Kaggle T4 timing of v0_b0 kit.
