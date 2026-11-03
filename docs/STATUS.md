# Coordinator status

Last update: 2026-11-04 01:50 IST.

## Phase / step
Phase 2 §3 steps 1-2 done (verdicts in EXPERIMENTS "Decisions"): recipe so far = p2_ref + `seen_corruptions_p: 0.9` (`configs/p2/p2_corr_p09.yaml`). Steps 3-4 running. Phase 3 + Phase 4 code merged (Phase 4 plumbing merge ed31dc0, GPU smoke OK). ~1 day ahead of PLAN §10.

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
- GPU queue `queues/p2_steps34.yaml` (started 01:48): p2_ema, p2_ep8, p2_ls05, p2_dp02, p2_ep20_lr1e4, p2_ls10, p2_drop05 (3 folds each, ~13.5 min; ep20 ~22). ETA ~03:30. Log `outputs/runs/queue_p2_steps34.log`. Restartable: rerun `DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset .venv/bin/python scripts/run_queue.py queues/p2_steps34.yaml`.
- **Laptop shutdown scheduled 04:00** (user request; `shutdown -c` cancels). power/control = on.

## Pending user decisions
- Approve audit grids: `outputs/audit/pair_masks.jpg` (31 masks, ~18 clean; PHASE4 §1 deviations recorded) and `outputs/audit/sbi_examples.jpg` (default prior, greyscale guard). `data/masks*`, `data/mask_prior.npy` stay uncommitted until approved (needed for p4 configs; reviewer noted fresh clones need them).

## Next 3 actions (4 Nov morning)
1. compare.py: p2_ema vs p2_corr_p09; ep8/ep20/ls/dp/drop vs p2_ema; record verdicts; write `configs/p2/p2_final.yaml`; run it on 5 folds (step 7).
2. Ask user to approve the two audit grids; rebase configs/p3 + configs/p4 onto p2_final; queue Phase 4 B0 ablations (3-fold) + PHASE3 step 0 (weight downloads, B4@320 batch probe) and step 1 LR sanity.
3. Dispatch: LOCO clusters (PHASE4 §6.3) + gradcam.py, PHASE5 ensemble.py/bundle.py/`--fold full` (opus).
