# Coordinator status

Last update: 2026-11-04 00:10 IST.

## Phase / step
Phase 2: §2 code done and committed (reproduction passed). §3 steps 1-2 running overnight. ~half a day ahead of PLAN §10.

## Running
- GPU queue `queues/p2_steps12.yaml` (7 runs x folds 0,1,2: p2_ref, p2_ref_seed43, p2_corr_p07, p2_corr_p09, p2_corr_weighted, p2_corr_cont, p2_extra). Started 23:20. Log `outputs/runs/queue_p2_steps12.log`, per-run `outputs/runs/<run>.log`, monitor `outputs/runs/gpu_monitor.csv`. ETA ~01:30-02:30. Restartable (`--skip_done`): rerun the same command.
- **Laptop shutdown scheduled 04:00 (user request, `shutdown -h 04:00`; cancel with `shutdown -c`).**
- Phase 3 model code committed (e124ce3). ConvNeXt-S layer_decay 0.93 (deviation, see PHASE3).
- Phase 4 generators committed (4801d29). Pair masks: only 31 accepted (~18 clean) of 187 pairs (most pairs are re-crops/zooms; many pairs.csv 'pairs' are different people). Deviations: affine ECC alignment, cc>=0.97 gate, constant-block exclusion, brightness match. SBI: default prior (centre 112,112; axes 49x63) because mask_prior spread too large; greyscale guard (spread<8, 5.8% of reals). Audit grids: outputs/audit/pair_masks.jpg, outputs/audit/sbi_examples.jpg (user approval pending; data/masks* uncommitted until then).
- Subagent (opus, git worktree): Phase 4 plumbing (mask head, SBIDataset, pair batching, sbi_only, pair/non-pair breakdown, configs/p4). Merge after review + GPU smoke.
- Epochs 12-16 s (GPU power-capped, throttle 0x4); queue ETA ~02:15.

## Pending user decisions
- Approve pair-mask grid (31 masks, deviations from PHASE4 §1) and SBI grid.
- none. (power/control set to `on` by user 23:3x, confirmed.)

## Next 3 actions (4 Nov morning)
1. Read queue log; compare.py: seed43 vs ref (noise floor -> threshold), p07/p09 vs ref, weighted/cont/extra vs ref; record verdicts.
2. Combined step-2 winner run, then step 3 (p2_ema, p2_ep8, p2_ep20_lr1e4) configs + queue.
3. Show user pair-mask + SBI audit grids; then mask head / SBIDataset / pair batching plumbing (opus, worktree). B4@320 batch probe + weight downloads when GPU is free (PHASE3 step 0).
