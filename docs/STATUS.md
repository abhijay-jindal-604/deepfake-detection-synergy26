# Coordinator status

Last update: 2026-11-09 22:30 IST (analysis + GPU agent).

## Phase / step
Phases 2, 3, 4 done; Phase 5 ensemble chosen and **submission v1_ens built** (local GPU). Feature freeze in effect. Details and numbers: `docs/EXPERIMENTS.md` (rows dated 9 Nov, "Phase 3 report" onwards).
- **Phase 3:** 5-fold selection B0 0.8878, ConvNeXt-S 0.8914, ViT-B 0.8907, B4@320 0.8800 (best clean/OOF 0.9009 but fragile to strong noise). Formal PHASE3 §4 no-go (no backbone +0.01 over B0); not debugged further (freeze); the value shows up as ensemble diversity.
- **Phase 4:** SBI (0.3, 0.6, +mask) REJECT; mask head and pair batching TIE → not adopted. SBI-only diagnostic 0.526: SBI pseudo-fakes don't transfer to this data. LOCO p2_final: held-out-cluster AUC 0.780 vs 0.889 seen (cluster 3 0.631); clusters are content/corruption clusters, not methods.
- **Phase 5:** ensemble = **p2_final (B0) + b4_320_final + vitb_final**, equal weights, T_m 1.129/1.002/0.984, T_ens 0.917, **threshold 0.4282**, **no flip TTA** (gain 0.000 < 0.003). OOF AUC **0.9074** vs best single 0.9009; J 0.9010 vs best single 0.8910. Families EfficientNet + ViT.

## Decision threshold
Δselection >= 0.006 on the 3-fold mean and > 0 on all 3 folds (noise floor 4 Nov). Use `scripts/compare.py A B --folds 0,1,2 --threshold 0.006`.

## Running
Nothing. GPU free.

## Submissions
- **v1_ens (current best):** `outputs/submissions/v1_ens/` = predictions.csv, model_weights.pt (410.5 MiB), ensemble.json/.md, inference_run{1,2}.log, validate.txt (ALL PASS), logs/ (full-run metrics, configs, train logs). Local RTX 5060 36-38 s wall, byte-identical re-run. **Not yet produced on a T4.**
  Rebuild: `python scripts/bundle.py --ensemble outputs/ensemble/ensemble.json --source full --out <dir>/model_weights.pt` → `python inference_script.py --test_dir <test> --weights <dir>/model_weights.pt --output <dir>/predictions.csv`.
- **v0_b0 (safety net):** `outputs/submissions/v0_b0/` (single B0, 13.8 s wall).
- **Kaggle T4 kit for v1:** `outputs/kaggle_kit_v1.zip` (468 MB; bundle + script + validator + t4_run.py + 3,000 test images + local CSV). Folder: `outputs/kaggle_kit_v1/`. Instructions: `scripts/kaggle/README.md`.

## Pending user decisions
- None blocking. Optional: if the T4 run is > 2.5 min, accept the cut (drop B4@320 → p2_final + vitb_final, OOF 0.9052 / J 0.8988) or keep v1 as long as it is < 5 min (20-pt band).

## Next actions (10 Nov)
1. **User: Kaggle T4 run** (`scripts/kaggle/README.md` steps B-D): upload `outputs/kaggle_kit_v1.zip` as a **private** dataset, import `scripts/kaggle/t4_timing.ipynb`, GPU T4, Internet on, Run All, download `t4_results.zip` into `outputs/kaggle/`. Check: sm_75 PASS, wall time (estimate ~1.6-2.4 min), validator PASS, re-run identical, label flips vs local.
2. If T4 OK: the T4 `predictions.csv` becomes the submitted one (PHASE5 §6.2); upload v1 to the Drive folder as the safety copy (PHASE5 §7, 11 Nov).
3. If T4 > 2.5 min: PHASE5 §5 cut order (no TTA already) → drop B4@320 (re-bundle `--member p2_final:full --member vitb_final:full` with the ensemble's T/threshold, or re-run `dfd.ensemble --runs p2_final vitb_final`) → `--batch_size 128`; re-time.
4. Phase 6: fill report TODO(results) placeholders from the EXPERIMENTS rows (Phase 3 table, Phase 4 verdicts, LOCO, ensemble), Grad-CAM, cross-device diff once the T4 CSV exists.

## Notes
- `run_queue.py` does not handle `--fold full`; use `scripts/p5_full.sh` (pattern) or `dfd.train --fold full` directly with nohup/setsid.
- Inference per member alone (local): B0 10.7 s (data-loading bound), B4@320 16.1 s, ViT-B 5.8 s; ensemble 31 s inference.
- cnxs_final/full exists (pruned from the ensemble; available as an alternate member).
- Don't redirect run_queue stdout to the queue's own log file.
