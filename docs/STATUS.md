# Coordinator status

Last update: 2026-11-09 16:05 IST (GPU build agent).

## Phase / step
Phase 2 COMPLETE (p2_final 5-fold OOF AUC 0.8964; recipe `configs/p2/p2_final.yaml`). Phase 5 code merged to main (f61c07a); CPU + GPU FP16 parity verified. **Safety-net submission v0_b0 exists** (`outputs/submissions/v0_b0/`, single B0 full-data retrain, validator ALL PASS). Phase 3 steps 0-1 done (LRs: cnxs 5e-5, b4_320 2e-4, vitb 5e-5; see EXPERIMENTS). Audit grids approved, Phase 4 data committed (f17e797). **Overnight GPU chain running** (Phase 3 step 2 + Phase 4 B0 ablations + LOCO + flip-TTA).

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
- **GPU chain `scripts/p3_step2.sh`** started 16:02, stdout `outputs/runs/p3_step2.stdout`, per-queue logs `outputs/runs/queue_<q>.log`, per-run `outputs/runs/<run>.log`. Order and ETA (local RTX 5060):
  1. `queues/p3_final.yaml`: cnxs_final 5 folds (~47 min, ~16:50) → b4_320_final (~62 min, ~17:55) → vitb_final (~50 min, ~18:45)
  2. `queues/p4_b0.yaml` (B0, folds 0,1,2): p4_sbi03 → p4_sbi_mask → p4_maskhead → p4_sbi06 → p4_pairbatch → p4_sbi_only (fold 0) (~70-80 min, ~20:05)
  3. `queues/loco_p2_final.yaml` (4 x fold 0, ~15-20 min, ~20:25)
  4. flip-TTA OOF + val-C (`dfd.evaluate --tta_flip`) for cnxs_final, b4_320_final, vitb_final, p2_final (20 folds, ~1-1.3 h) → **chain done ~21:30 (±45 min)**
- **Resume** (skips finished folds and finished TTA evals): `DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset nohup setsid scripts/p3_step2.sh > outputs/runs/p3_step2.stdout 2>&1 < /dev/null &` (check nothing is running first: `pgrep -af dfd.train`). Only one GPU job at a time.
- After: Phase 3 summary (`scripts/compare.py`, OOF correlations), P4 verdicts vs p2_final (`compare.py p2_final p4_x --folds 0,1,2 --threshold 0.006` + LOCO/pair rule), `python -m dfd.loco report --names p2_final` (rerun `dfd.loco configs` with the SBI config for the 2nd LOCO recipe if SBI looks promising).

## Safety net (v0_b0)
`outputs/submissions/v0_b0/predictions.csv` + `model_weights.pt` (15.6 MiB) + run logs + `validate.txt`. Rebuild: `python scripts/bundle.py --member p2_final:full --out <dir>/model_weights.pt`, then `python inference_script.py --test_dir <test> --weights <dir>/model_weights.pt --output <dir>/predictions.csv`. Local GPU 13.8 s wall, byte-identical re-run. Not yet produced on a T4 (Kaggle kit: `scripts/kaggle/`).

## Notes
- `run_queue.py` does not handle `--fold full`; launch full-data retrains with `nohup .venv/bin/python -m dfd.train --config <cfg> --fold full --run_name <run>` directly.
- Inference throughput (B0, 3k imgs): ~5 s fixed CUDA/worker warm-up + ~550 img/s; data loading (4 workers) is the bottleneck for B0, not the GPU.
- Don't redirect run_queue stdout to the queue's own log file (it writes `outputs/runs/queue_<name>.log` itself; lines get duplicated).

## Pending user decisions
- (none from this agent; audit grids approved 9 Nov.)
- Watch: on fold 0 no big backbone beats B0 p2_final (PHASE3 §4 no-go needs +0.01 on 5 folds); decide after the 5-fold results.

## Next actions
1. Morning 10 Nov: check `outputs/runs/p3_step2.stdout`; Phase 3 summary + P4 verdicts + LOCO report; ensemble pool → `dfd.ensemble` (with/without `--variant tta_flip`).
2. Full-data retrains of the chosen members (`--fold full`, nohup), bundle, inference, Kaggle T4 timing (`scripts/kaggle/`).
