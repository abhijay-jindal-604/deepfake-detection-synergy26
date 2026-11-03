# Phase 6: Deliverables, report, TF export, submission (target: 11-14 Nov 2026)

Scope: turn the working system into the scored package. That's 10 documentation points, +2 Grad-CAM, +1 TF export, the public GitHub repo, the Drive folder plus zip, and a fresh-clone reproduction test. **Submit on Unstop by the evening of 14 Nov** (the round window opens 12 Nov 11:59 PM and reportedly closes 15 Nov 12:26 AM).

Parent plan: `docs/PLAN.md` §8. Previous: `docs/PHASE5.md`. Decisions from the user (3 Nov): **public GitHub repo, with no Claude co-author attribution**; **TF export: committed**; Kaggle account to be set up on 4 Nov.

---

## 1. Figures and logs (11 Nov, ~3 h, light GPU)

| artefact | tool | content |
|---|---|---|
| Grad-CAM grids (+2 bonus) | `src/dfd/gradcam.py` (started in Phase 4; manual hooks, ~80 lines, no new package) | For each ensemble member: a 4x4 grid of **validation** images (fold-0 model on fold-0 val; never test images) as TP / TN / FP / FN, image + heatmap overlay. Plus one grid of corrupted val images (snow, patches, pixelate) showing attention staying on the face. Plus the Phase 4 before/after grid. CNNs: Grad-CAM on the last conv stage. ViT: Grad-CAM on the last block's patch tokens reshaped to 14x14 (the standard "reshape transform"). |
| loss/AUC curves | `scripts/plot_curves.py` (exists) | 5 folds per final member (train loss, val loss, val AUC raw vs EMA) + the full-data runs' train loss on top of the fold curves (the §3 check in PHASE5) |
| val-C chart | `scripts/plot_valc.py` (new, ~40 lines) | per-corruption AUC bars: B0 baseline vs Phase 2 recipe vs final ensemble; held-out corruptions marked |
| ablation table | `scripts/compare.py --markdown` | from EXPERIMENTS.md: every adopted/rejected change with Δ and verdict |
| logs | copy `metrics.csv`, `eval.txt`, `oof_summary.txt` of the final runs + `EXPERIMENTS.md` | optional deliverable 7 |

All figures go to `outputs/figures/` (git-ignored); the ones used in the report are copied to `docs/report/fig/` (committed, small PNGs).

## 2. Report (≥ 4 pages; 12-13 Nov, ~6-8 h total)

Source: `docs/report/report.md` → `scripts/build_report.py` (python-markdown, already installed via tensorboard → HTML with print CSS → `google-chrome --headless --print-to-pdf`). No pandoc on this machine, and none needed.
Target length: **6-8 pages + appendix**. The first page is a standalone executive summary, which also covers the Unstop "1-2 page report" reading.

### Outline mapped to the 10-point documentation rubric
| § | section | rubric item (pts) | content and evidence |
|---|---|---|---|
| 0 | Executive summary (1 page) | — | Approach in 5 bullets; final OOF AUC, val-C, T4 time; one architecture figure; one results table |
| 1 | Problem and approach | problem/approach (2) | Task, metric (AUC), constraints. **Data audit**: in-the-wild swaps, corruptions on both classes, single JPEG quant table, real/fake pairs → grouped CV (with a pair example image). Strategy: robustness first, method-independent cues (SBI), diverse ensemble. |
| 2 | Architecture | architecture (2) | Ensemble diagram; per member: backbone, pretraining (IN-1k/22k), input size, head (zero-init), aux mask head (training only), params, FP16 inference; how logits are combined (temperature, weights) |
| 3 | Hyperparameters and their justification | hyperparameter justification (2) | Table: parameter, value, *why*, *evidence* (Δ from our ablations: corruption p, EMA, epochs, LR/LLRD per backbone, label smoothing, drop-path, mixup = no, CutMix = no and why, SBI ratio, mask weight, TTA, threshold). Negative results included. |
| 4 | Training pipeline | training pipeline (2) | Flow figure: grouped folds → augmentation (seen corruptions, extra aug, held-out deny-list) → SBI → loss → AdamW + LLRD + warmup/cosine + EMA → fixed-epoch last checkpoint → full-data retrain → ensemble selection → threshold. Loss curves figure. Decision protocol (paired 3-fold, noise floor). |
| 5 | Results | (supports all) | OOF AUC / F1 / acc / threshold on 5-fold and on the organiser-style fold-0 80/20 split; val-C table (seen / held-out / worst); ablations; ensemble vs best single; LOCO (unseen-method proxy); T4 timing; Grad-CAM figures |
| 6 | Limitations | limitations (1) | val-C is a guess at the hidden variants; LOCO clusters ≠ true generation methods; only ~150 pair masks; full-data models have no own validation; label noise; FP16 cross-GPU differences; CLIP-style backbones excluded by rule reading; single seed |
| 7 | Reproducibility | reproducibility (1) | Env (Python 3.10, pinned versions, GPU), seeds, exact commands (folds → train → ensemble → bundle → inference), runtimes per step, determinism tests (double-run, cross-device Δ), SHA-256 of `model_weights.pt` |
| A | Appendix | — | Full val-C table per member, configs, glossary |

Writing order: §1, §4, §3 on 12 Nov (they draw on finished work), §2, §5, §6, §7 + summary on 13 Nov, then build the PDF and check the page count and figures.

## 3. README and requirements (12 Nov, ~2 h)

`README.md` (repo root) contains:
1. One-paragraph overview + result numbers.
2. **Setup:** `uv venv --python 3.10 .venv && uv pip install -r requirements.txt` (training, full pins, cu128 index) or `requirements-infer.txt` (inference only).
3. **Inference (what the organisers run):** `python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv`, plus the expected T4 time.
4. **Full reproduction:** `export DF_DATA_ROOT=...`; `python -m dfd.groups` (optional; `data/folds.csv` is committed); `python -m dfd.masks`; `python -m dfd.train --config configs/final/<member>.yaml --fold all` (CV) and `--fold full` (final); `python -m dfd.ensemble`; `python scripts/bundle.py`; then inference. Expected time per step on an 8 GB GPU.
5. Repo map; where results are logged; licence note (our code only; the dataset isn't redistributed).

`requirements-infer.txt`: python 3.10; `torch==2.11.0`, `torchvision==0.26.0` (timm imports it), `timm==1.0.30`, `numpy==2.2.6`, `pandas==2.3.3`, `opencv-python-headless==5.0.0.93`, `pillow`, `safetensors`, `huggingface-hub`, `pyyaml`, with versions copied from `requirements.txt`, plus a cu128 `--extra-index-url` (or whatever Phase 5 §5 had to pin for sm_75).

## 4. TF export (+1; committed; **13 Nov, time-boxed to 5 h**)

The bonus asks for "export for both PyTorch and TF". Plan: export the *same* ensemble to a TensorFlow SavedModel and ship a TF inference script with the same CLI.

1. **Separate env** `.venv-tf` (Python 3.10, via uv): `tensorflow` (latest version with Python 3.10 wheels at that time, pinned), `onnx`, `onnxruntime`, `onnx2tf` (and its helpers `onnx-graphsurgeon`, `sng4onnx`), pinned in `requirements-tf.txt`. Kept separate so TF can never break the PyTorch env. *(This is the only new install in the plan; it happens on 13 Nov, not before.)*
2. `scripts/export_onnx.py`: each member in FP32 → `torch.onnx.export` (opset 17, dynamic batch, input = float 0-255 NCHW, since normalisation and upsampling are inside the model) → check against onnxruntime: max |Δlogit| < 1e-4 on 64 val images.
3. `onnx2tf -i member.onnx -o tf/member_k` (it converts NCHW → NHWC). Check against onnxruntime: |Δlogit| < 1e-3.
4. `scripts/build_tf_ensemble.py`: a `tf.Module` that loads the member SavedModels, applies the temperatures/weights and the sigmoid → one SavedModel `tf_model/` + `ensemble.json`.
5. `inference_script_tf.py`: same CLI (`--weights` = the SavedModel dir), same I/O as the PyTorch script.
6. **Parity:** TF vs PyTorch on 512 val images: max |Δconfidence| < 1e-3, identical AUC to 4 decimals, 0 label flips. Report it, and the TF timing, in the report's reproducibility section.

Known risk points: ViT attention / GELU and ConvNeXt's channels-last LayerNorm are the usual onnx2tf trouble spots; flags `-cotof` (output-check) and opset 17 vs 18 are the first knobs. **Fallback ladder:** full ensemble → the members that convert (document which) → best single member. **Hard stop at 18:00 on 13 Nov**: whatever works by then ships; if nothing works, the report mentions the attempt and we drop the bonus. The deadline is worth more than +1.

## 5. GitHub (public, no Claude co-author) (12 Nov, ~1 h)

1. **History attribution (needs your explicit OK at that time; it rewrites history and can't be undone once pushed):** the 8 Phase-1 commits end with `Co-Authored-By: Claude …`. Before the *first* push, strip those lines: `git filter-branch --msg-filter "sed '/^Co-Authored-By: Claude/d'" -- --all` (built into git; no install), on a backup branch first, then `git log` check. From 3 Nov on, commits are written without the trailer.
2. **Ask you at that time:** keep `CLAUDE.md` in the public repo, or remove it from the tracked files (it's the project-notes file the assistant reads)? It's your call.
3. Check nothing large or private is tracked: `resources/`, `outputs/`, `.venv*/`, `*.pt` are ignored; `data/*.csv` (train IDs, folds, pairs) and `data/masks/` are fine to publish (derived from train metadata); no test images and no test IDs in the repo.
4. `gh repo create <name> --public --source . --push`, then add the URL to the README and report.
5. When to go public: the repo shows our approach to other teams once it's public. Default plan: push on **12 Nov** (after the scrub), with timestamped commits as proof of authorship. Pushing earlier is only if you want it.

## 6. Packaging (14 Nov morning, ~2 h)

`scripts/package.py` builds `outputs/submission/`:
```
abhijayjindal604_Synergy26_Deepfake/            (Google Drive folder, "Anyone with the link: Viewer")
├── predictions.csv              (from the T4 run, validator passed)
├── model_weights.pt             (bundle; SHA-256 in README/report)
├── inference_script.py
├── requirements.txt             (= requirements-infer.txt; the template expects this name)
├── report.pdf                   (≥ 4 pages; page 1 = summary)
├── source_code.zip              (git archive HEAD, includes README.md) + GitHub link in README.txt
├── logs/                        (metrics.csv per run, curves PNGs, EXPERIMENTS.md)
├── gradcam/                     (figures)
├── tensorflow/                  (tf_model/, inference_script_tf.py, requirements-tf.txt)
└── abhijayjindal604_submission.zip   (predictions.csv, inference_script.py, model_weights.pt,
                                       requirements.txt, report.pdf)
```
- If Unstop's upload limit is smaller than the zip (~320 MB with weights), upload `abhijayjindal604_submission.txt` with the **public Drive folder link** instead (the brief allows this).
- **Public check:** open the Drive link in a private/incognito window while logged out; every file must open/download. Non-public = disqualified.
- Check the team name spelling against the Unstop registration ("abhijayjindal604").

## 7. Fresh-clone reproduction test (14 Nov, ~1.5 h)

1. `git clone <public GitHub URL>` into a scratch directory; new `uv venv --python 3.10`; `pip install -r requirements-infer.txt`.
2. Download `model_weights.pt` **from the public Drive link** (as the organisers would).
3. Run `inference_script.py` on `test/images/` → `validate_predictions.py` → compare with the submitted CSV: local GPU vs T4 CSV within 1e-3, label flips listed.
4. On Kaggle (T4, fresh session, internet off for the run): same steps; the CSV must be **byte-identical** to the submitted one (same GPU class + deterministic script).
5. Training-code smoke from the clone: `python -m dfd.train --config configs/final/<member>.yaml --fold 0 --limit 512 --epochs 1 --valc_limit 64 --no_log` runs. This proves the source code is complete.

## 8. Calendar (Phase 6)
| day | work |
|---|---|
| 11 Nov (after Phase 5) | Grad-CAM, curves, val-C chart, ablation table |
| 12 Nov | Report §1, §3, §4; README; requirements-infer; **history scrub (with your OK) + public GitHub push** |
| 13 Nov | **TF export (09:00-18:00 hard stop)**; report §2, §5-7 + summary; build PDF |
| 14 Nov | Package; Drive upload + public check; fresh-clone + Kaggle reproduction; **Unstop submission by ~18:00**; final commit/tag `v1.0-submission` |
| 15 Nov | Buffer only (fixes, re-upload if something breaks; latest Drive upload counts) |

## 9. Risks
| risk | mitigation |
|---|---|
| Drive folder not public → disqualification | Incognito check; second check after the final upload |
| TF export eats the day | 18:00 hard stop; fallback ladder |
| Report too thin on justification | §3 evidence table comes straight from EXPERIMENTS.md verdicts; written while the results are fresh |
| Upload size limit on Unstop | txt-with-link variant prepared |
| Public repo copied by others | Our commit timestamps predate theirs; push after 12 Nov |
| Last-minute change breaks reproducibility | Any change after 14 Nov morning → rerun the validator + double-run + Kaggle run before re-upload |

## Phase 6 exit criteria
- [ ] report.pdf ≥ 4 pages, every rubric item has its own section; Grad-CAM included.
- [ ] README with exact reproduction commands; `requirements-infer.txt`.
- [ ] Public GitHub repo without Claude co-author lines (history scrub approved by the user); URL in README/report.
- [ ] TF SavedModel + script + parity numbers, or a documented, time-boxed attempt.
- [ ] Drive folder public (incognito-checked); zip/txt uploaded on Unstop; fresh-clone + Kaggle re-run match.
