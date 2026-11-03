# Coordinator status

Last update: 2026-11-04 02:40 IST (handover before 04:00 shutdown).

## Phase / step
Phase 2 §3 steps 1-2 done (verdicts in EXPERIMENTS "Decisions"): recipe so far = p2_ref + `seen_corruptions_p: 0.9` (`configs/p2/p2_corr_p09.yaml`). Step 3: EMA adopted, 12 epochs kept (ep8 rejected). Steps 3-4 running. Phase 3 + Phase 4 code merged (Phase 4 plumbing merge ed31dc0, GPU smoke OK). ~1 day ahead of PLAN §10.

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
- GPU queue `queues/p2_steps34.yaml` (started 01:48): p2_ema, p2_ep8, p2_ls05, p2_dp02, p2_ep20_lr1e4, p2_ls10, p2_drop05 (3 folds each, ~13.5 min; ep20 ~22). ETA ~03:30. Log `outputs/runs/queue_p2_steps34.log`. Restartable: rerun `DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset .venv/bin/python scripts/run_queue.py queues/p2_steps34.yaml`.
- **Laptop shutdown scheduled 04:00** (user request; `shutdown -c` cancels). power/control = on.

- Phase 5 code on branch `worktree-agent-a98d6a22ba82f3084` (ensemble.py, bundle.py, inference_script.py, parity_test.py, `--fold full`, `--tta_flip`); parity vs dfd.models exact in FP32; reviewed (merge verdict) + robustness fixes (unreadable image -> 0.5 row; DataLoader worker fallback), branch head 575df24. **Merge in the morning before any new GPU queue.** CUDA/FP16 inference path untested (needs GPU + Kaggle T4). `scripts/validate_predictions.py` committed on main. Note: BF16 OOF probs differ from FP32 inference by up to 0.19 per image (AUC +7e-4) — fine for decisions.
- Incident 02:00: Phase 5 agent ran ~40 s of inference on the GPU during the queue; no effect on the run; later briefs force CUDA_VISIBLE_DEVICES="".
- Queue cut expected: ls10/drop05 likely cut by the 04:00 shutdown (lowest in PHASE2 cut order); rerun the queue command to finish them.

## Pending user decisions
- Approve audit grids: `outputs/audit/pair_masks.jpg` (31 masks, ~18 clean; PHASE4 §1 deviations recorded) and `outputs/audit/sbi_examples.jpg` (default prior, greyscale guard). `data/masks*`, `data/mask_prior.npy` stay uncommitted until approved (needed for p4 configs; reviewer noted fresh clones need them).

## Next actions (4 Nov morning)
1. Check `outputs/runs/queue_p2_steps34.log` (shutdown likely cut ls10/drop05 mid-run): rerun the queue command to finish them (restartable). Merge `worktree-agent-a98d6a22ba82f3084`, GPU smoke of inference_script.py (FP16 path) on train images.
2. compare.py (threshold 0.006): ls05/dp02/ep20_lr1e4/ls10/drop05 vs p2_ema; record verdicts; write `configs/p2/p2_final.yaml`; run it on 5 folds (step 7) -> new reference.
3. Ask user: approve the two audit grids. Then rebase configs/p3 + configs/p4 on p2_final; queue PHASE3 step 0 (weight downloads, B4@320 batch probe) + step 1 LR sanity, then Phase 4 B0 ablations.
4. Dispatch: LOCO clusters (PHASE4 §6.3) + gradcam.py; requirements-infer.txt; Kaggle T4 timing plan (user does the upload).
