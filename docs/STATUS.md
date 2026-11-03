# Coordinator status

Last update: 2026-11-04 00:05 IST.

## Phase / step
Phase 2: §2 code done and committed (reproduction passed). §3 steps 1-2 running overnight. ~half a day ahead of PLAN §10.

## Running
- GPU queue `queues/p2_steps12.yaml` (7 runs x folds 0,1,2: p2_ref, p2_ref_seed43, p2_corr_p07, p2_corr_p09, p2_corr_weighted, p2_corr_cont, p2_extra). Started 23:20. Log `outputs/runs/queue_p2_steps12.log`, per-run `outputs/runs/<run>.log`, monitor `outputs/runs/gpu_monitor.csv`. ETA ~01:30-02:30. Restartable (`--skip_done`): rerun the same command.
- **Laptop shutdown scheduled 04:00 (user request, `shutdown -h 04:00`; cancel with `shutdown -c`).**
- Phase 3 model code committed (e124ce3). ConvNeXt-S layer_decay 0.93 (deviation, see PHASE3).
- Subagents (CPU only): PHASE4 §1 `masks.py` (pair masks + audit grid), PHASE4 §3.1 `sbi.py` (SBI generator + audit grid). Both audits need user approval before any training uses them.
- Epochs 12-16 s (GPU power-capped, throttle 0x4); queue ETA ~02:15.

## Pending user decisions
- none. (power/control set to `on` by user 23:3x, confirmed.)

## Next 3 actions (4 Nov morning)
1. Read queue log; compare.py: seed43 vs ref (noise floor -> threshold), p07/p09 vs ref, weighted/cont/extra vs ref; record verdicts.
2. Combined step-2 winner run, then step 3 (p2_ema, p2_ep8, p2_ep20_lr1e4) configs + queue.
3. Show user pair-mask + SBI audit grids; then mask head / SBIDataset / pair batching plumbing (opus, worktree). B4@320 batch probe + weight downloads when GPU is free (PHASE3 step 0).
