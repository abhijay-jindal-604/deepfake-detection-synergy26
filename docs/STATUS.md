# Coordinator status

Last update: 2026-11-04 03:40 IST (handover before 04:00 shutdown).

## Phase / step
Phase 2 COMPLETE: p2_final 5-fold OOF AUC 0.8964 (baseline 0.8715). Steps 1-6 done (all verdicts in EXPERIMENTS Decisions). **Recipe = `configs/p2/p2_final.yaml`**: B0 + seen corruptions p0.9 + EMA + drop_path 0.2 + LS 0.10, 12 epochs, last EMA ckpt. Step 7 done 03:53. Phase 3 + 4 code merged; Phase 5 code on branch (reviewed). ~1 day ahead of PLAN §10.

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
- GPU queue `queues/p2_final.yaml` (p2_final, folds all; started 03:38, ~11 min/fold when CPU is quiet). The 04:00 shutdown will cut it: **rerun `DF_DATA_ROOT=resources/extracted/dataset/hackathon_dataset .venv/bin/python scripts/run_queue.py queues/p2_final.yaml`** (skips finished folds, then writes oof_all + summary).
- Finding: CV runs are CPU-bound when subagents run CPU-heavy checks (20 min vs 11 min per 3-fold run). Keep CPU-heavy subagent work light during queues.
- **Laptop shutdown scheduled 04:00** (user request; `shutdown -c` cancels). power/control = on.

- Phase 5 code on branch `worktree-agent-a98d6a22ba82f3084` (ensemble.py, bundle.py, inference_script.py, parity_test.py, `--fold full`, `--tta_flip`); parity vs dfd.models exact in FP32; reviewed (merge verdict) + robustness fixes (unreadable image -> 0.5 row; DataLoader worker fallback), branch head 575df24. **Merge in the morning before any new GPU queue.** CUDA/FP16 inference path untested (needs GPU + Kaggle T4). `scripts/validate_predictions.py` committed on main. Note: BF16 OOF probs differ from FP32 inference by up to 0.19 per image (AUC +7e-4) — fine for decisions.
- Incident 02:00: Phase 5 agent ran ~40 s of inference on the GPU during the queue; no effect on the run; later briefs force CUDA_VISIBLE_DEVICES="".

## Pending user decisions
- Approve audit grids: `outputs/audit/pair_masks.jpg` (31 masks, ~18 clean; PHASE4 §1 deviations recorded) and `outputs/audit/sbi_examples.jpg` (default prior, greyscale guard). `data/masks*`, `data/mask_prior.npy` stay uncommitted until approved (needed for p4 configs; reviewer noted fresh clones need them).

## Next actions (4 Nov morning)
1. (p2_final 5-fold done.) Merge `worktree-agent-a98d6a22ba82f3084` (Phase 5) after the queue; GPU smoke of inference_script.py FP16 path on train images.
2. Ask user: approve the two audit grids. Rebase configs/p3 + configs/p4 onto `../p2/p2_final.yaml`; queue PHASE3 step 0 (weight downloads, B4@320 batch probe) + step 1 LR sanity (fold 0), then Phase 4 B0 ablations (3-fold, compare vs p2_final).
3. Dispatch: LOCO clusters (PHASE4 §6.3) + gradcam.py; requirements-infer.txt; Kaggle T4 timing plan (user does the upload).
