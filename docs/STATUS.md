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
- **v1_ens (current best):** `outputs/submissions/v1_ens/` = predictions.csv, model_weights.pt (410.5 MiB), ensemble.json/.md, inference_run{1,2}.log, validate.txt (ALL PASS), logs/ (full-run metrics, configs, train logs). Local RTX 5060 36-38 s wall, byte-identical re-run. **T4 verified 12 Nov (Kaggle): cold 72-111 s over two sessions (20-pt band; report in outputs/kaggle/t4_out/), sm_75 PASS, 0 label flips vs local, max |Δconf| 1.1e-3. Local CSV kept as the submitted one; uploaded to Drive folder `abhijayjindal604_deepfake_submission` (private; weights + report still to add by hand).**
  Rebuild: `python scripts/bundle.py --ensemble outputs/ensemble/ensemble.json --source full --out <dir>/model_weights.pt` → `python inference_script.py --test_dir <test> --weights <dir>/model_weights.pt --output <dir>/predictions.csv`.
- **v0_b0 (safety net):** `outputs/submissions/v0_b0/` (single B0, 13.8 s wall).
- **Kaggle T4 kit for v1:** `outputs/kaggle_kit_v1.zip` (468 MB; bundle + script + validator + t4_run.py + 3,000 test images + local CSV). Folder: `outputs/kaggle_kit_v1/`. Instructions: `scripts/kaggle/README.md`.

## Pending user decisions
- OK to make the Drive folder public ("Anyone with the link can view"); required before submitting.
- OK to scrub the 8 Co-Authored-By trailers from git history before the public GitHub push.
- User must drag `outputs/submissions/v1_ens/model_weights.pt` (410 MB, sha256 ab8d0c7a...) into the Drive folder by hand (connector limit 10 MB).

## Next actions (from 12 Nov evening): two parallel sessions
Ownership, to avoid collisions: **Session A (report/submission)** owns `docs/report/`, `README.md`, Drive, GitHub, TF export, packaging.
**Session B (improvement)** owns new `configs/p7/`, new runs under `outputs/runs/`, and appends rows to `docs/EXPERIMENTS.md`.
Both: `git add` only your own paths; GPU is shared (B's training queue has priority; A's Grad-CAM/curves are short jobs, run them between B's runs or on CPU).
v1_ens stays the submission unless B produces a verified v2 (gate below) by **13 Nov 20:00 IST**.

**Session A:** fill the 22 `TODO(results)` in `docs/report/report.md` from EXPERIMENTS rows; figures (Grad-CAM TP/TN/FP/FN grid per member, loss/AUC curves, val-C chart, architecture diagram) into `docs/report/fig/`; build the PDF (`scripts/build_report.py`, >= 4 pages); TF export 13 Nov time-boxed to 18:00; remove stale `.claude/worktrees/`; history scrub + public GitHub push (after user OK); 14 Nov: zip + Drive (report, final files), public-link check in incognito, submit on Unstop by ~18:00, tag `v1.0-submission`.

_Session A progress (12 Nov):_ done: report TODOs filled + Grad-CAM paragraph + measured GPU time; figures in `docs/report/fig/` (`scripts/make_figures.py`); full report PDF 19 pp and 2-page `docs/report/summary.md` -> `outputs/report/report_summary.pdf` (for the zip); stale worktrees removed (P4 smoke runs archived in `outputs/archive/wt_a689_p4_runs/`); history scrubbed of Claude trailers (backup `outputs/archive/pre_scrub_2026-11-12.bundle`) and pushed public: https://github.com/abhijay-jindal-604/deepfake-detection-synergy26 (`origin`). Remaining markers: TODO(tf) x2, TODO(tag). Next: TF export 13 Nov (stop 18:00); 14 Nov packaging.

**Session B:** B0 screen on folds 0,1,2 vs p2_final: `p7_mixup` (mixup, untried; PHASE2 cut) and `p7_res288` (B0 input 288, untried). Gate: `scripts/compare.py p2_final <run> --folds 0,1,2 --threshold 0.006` (Δselection >= 0.006 and > 0 on all 3 folds). If a winner: port to b4_320/vitb (5 folds) → full retrains → `dfd.ensemble` → OOF must beat 0.9074 → `scripts/bundle.py` → local validator + double run → new Kaggle kit + T4 run (<= 2.5 min cold) → hand the v2 folder to session A. If no winner: record REJECT rows and stop.

_Session B progress (12 Nov 23:35 IST):_ P7 screen launched detached (PID 492922, `configs/p7/run_screen.sh`): `p7_mixup` (mixup alpha 0.2, p 0.5) then `p7_res288` (B0, `model.input_size` 288, batch size 64 unchanged), both on B0 folds 0,1,2 vs p2_final.

_Session B verdict (13 Nov):_ P7 screen finished, no errors. p7_mixup mean Δselection +0.0012 (f0 −0.0005, f1 +0.0061, f2 −0.0019) and p7_res288 +0.0025 (+0.0024, +0.0048, +0.0003): both below the 0.006 gate, so both REJECT (TIE). Next step: none to port; v1 stays the submission (details in `docs/EXPERIMENTS.md`).
- Both smoke tests passed; res288 peaked at about 4.8 GB of 8 GB.
- Logs: `outputs/runs/p7_screen.log`, `p7_mixup.log`, `p7_res288.log`. Expected finish about 00:15-00:35 IST on 13 Nov.
- Next: `compare.py --threshold 0.006`; a run passes only with Δselection >= 0.006 and improvement on all 3 folds. If neither passes, v1 stays the submission. Deadline for a verified v2: 13 Nov 20:00 IST.

## Notes
- `run_queue.py` does not handle `--fold full`; use `scripts/p5_full.sh` (pattern) or `dfd.train --fold full` directly with nohup/setsid.
- Inference per member alone (local): B0 10.7 s (data-loading bound), B4@320 16.1 s, ViT-B 5.8 s; ensemble 31 s inference.
- cnxs_final/full exists (pruned from the ensemble; available as an alternate member).
- Don't redirect run_queue stdout to the queue's own log file.

_Session B follow-up (13 Nov):_ p7_res288 completed all 5 folds + full retrain (no errors in logs). 5-fold compare vs p2_final: mean Δselection +0.0021 (f3 −0.0001), held-out +0.0037, worst +0.0232 -> TIE. Ensemble swap (p7_res288 for p2_final, same method as v1): OOF AUC 0.9072 vs v1 0.9074; 4-member 0.9076 (pruned by the search rule) -> both REJECT, v1_ens stays (`outputs/ensemble_v2c/`).

_Drive + repro (13 Nov morning, coordinator session):_ fresh clone of c3b5627 + organiser command reproduces v1 predictions.csv **byte-identical** (43.9 s local, validator ALL PASS); standalone dir with only inference_script.py + weights also identical; no hardcoded paths. Drive folder now has predictions.csv, inference_script.py, requirements(.txt, -infer.txt), README.md, source_code.zip (HEAD c3b5627), training_logs.zip, report.pdf (2-page summary), technical_report.pdf (19 pp, still 3 TODO(tf)/TODO(tag) lines: re-upload after TF export), gradcam.zip, github_link.txt, abhijayjindal604_submission.zip (no weights; README_weights.txt points to Drive). Missing: model_weights.pt (user, by hand). Folder still PRIVATE (needs user OK). Session A: don't upload to Drive; hand the final PDF to the coordinator session.

_Submitted (14 Nov):_ **DONE.** Drive folder public (anyoneWithLink reader), checked in incognito by the user; Unstop submission made with `outputs/submissions/abhijayjindal604_submission.txt` (Drive link). Drive: report.pdf = 2-page summary, technical_report.pdf = full 19 pp report (0 TODOs), model_weights.pt size matches the local bundle; the old technical_report.pdf and the stale submission zip were moved to the trash. Report additions: group-bootstrap 95% CI of the ensemble OOF AUC 0.8997-0.9145 (`scripts/bootstrap_ci.py`), harsh-val-C note, no-label-pruning limitation, PyTorch-only (no TF export). Repo tagged `v1.0-submission`. Drive `source_code.zip` is from c3b5627 and was left unchanged after submission (the GitHub tag is the reference).
