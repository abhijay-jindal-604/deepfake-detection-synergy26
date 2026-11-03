# Deepfake ML Hackathon (Synergy '26, IIIT Bangalore)

Status: **Phase 1 done (3 Nov).** Env, grouped folds, val-C, B0 baseline + 5-fold OOF; results in `docs/EXPERIMENTS.md`. Phases 2-6 planned: `docs/PHASE2.md`-`docs/PHASE6.md`, calendar in `docs/PLAN.md` §10. User decisions (3 Nov): public GitHub repo with **no Claude co-author lines in commits**; TF export committed (13 Nov, time-boxed); Kaggle T4 for timing. Run code with `.venv/bin/python -m dfd.<module>` and `DF_DATA_ROOT` set. Full brief: `brief.md`.

## Goal and constraints
Train a binary classifier separating **real (label 0)** from **deepfake (label 1)** 224x224 face images; output P(Fake) for **3,000 test images**. Team 1-4 (registered solo as "abhijayjindal604"). Prize pool Rs 20,000 (10k / 6k / 4k).
- Train **only on the provided dataset**; no external datasets, no deepfake-specific pre-trained detectors. Test images may not be used for training in any way (supervised or unsupervised).
- Pre-trained **ImageNet** architectures (EfficientNet, ViT, ResNet, ...) are allowed, fine-tuned on provided data only. Augmentation of provided images allowed.
- Framework: **Python 3.10**, **PyTorch 2.x or TensorFlow 2.x only**. Max model size 5 GB. Inference must finish within **60 min on an NVIDIA T4**.
- Fully reproducible; hardcoded predictions or hardcoded paths = failure/disqualification.

## Key dates (IST)
- 30 Oct 2026: registration closed.
- 12 Nov 2026, 11:59 PM -> 15 Nov 2026, 12:26 AM: submission round window on Unstop (as listed).
- **15 Nov 2026, 11:59 PM: final submission deadline** (time TBA). The Unstop round window reportedly ends 12:26 AM on 15 Nov, so **treat end of 14 Nov as the safe deadline**.
- One final submission per team; the Drive folder may be updated until the deadline, latest upload counts (tiebreak 4: earlier last upload).

## Deliverables and submission format
**Sources conflict (see Open questions).** Dataset README (latest, dated 30 Oct) says upload to a single Google Drive folder, "Anyone with the link can view":
1. `predictions.csv` (required)
2. Model weights (`.pt`/`.pth`, or `.h5`/SavedModel for TF) (required)
3. `inference_script.py` (required)
4. `requirements.txt` or `environment.yml` (required)
5. Source code (GitHub link or zipped folder with `README.md`) (required)
6. Technical report, PDF or Markdown, **min 4 pages** (required)
7. Training logs / loss curves (optional)
8. Grad-CAM / attention visualisations (optional, bonus)

The Unstop round text (per brief.md) instead says a single zip `"TeamName"_submission.zip` with predictions.csv, inference_script.py, weights, requirements.txt, and a **1-2 page** report.pdf; if too large, a txt file containing a **public** Drive folder link. Non-public folders are disqualified. Safest: satisfy both (Drive folder containing every item above plus the zip, report >= 4 pages).

### predictions.csv (exact)
Header `image_id,label,confidence`. One row per test image, all 3,000 rows.
- `image_id`: string, filename **without** `.jpg` (UUID), matching the test set exactly.
- `label`: int 0 or 1 (0 = Real, 1 = Fake).
- `confidence`: float in [0.0, 1.0] = **probability the image is FAKE**.
- Missing IDs are scored incorrect. Do not submit `sample_submission.csv` as-is (it has placeholder IDs).

### inference_script.py interface (exact)
```
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
```
- argparse args: `--test_dir` (required), `--weights` (required), `--output` (default `predictions.csv`). Template also adds optional `--batch_size` (default 64) and `--threshold` (default 0.5).
- Processes ALL images in `--test_dir` (template globs `*.jpg`, `*.jpeg`, `*.png`, sorted); `image_id = path.stem`.
- Template structure: `TestDataset`, `build_model()` (placeholder EfficientNet-B4 with 1-logit head, `weights=None`), `load_model()` (handles plain state_dict, `{"state_dict": ...}`, or full pickled model), `get_inference_transforms()` (Resize 224, ToTensor, ImageNet mean/std), `run_inference()` (sigmoid on a single logit -> confidence), validation asserts, then `to_csv(index=False)`.
- Template imports `pandas`, `numpy`, `torch`, `torchvision`, `PIL`; `DEVICE` = cuda if available. Organisers **re-run this script** to verify; mismatch with submitted predictions.csv costs -5 pts.
- **Template bug (verified with `compile()`):** `main()` uses `BATCH_SIZE`/`THRESHOLD` as argparse defaults before `global BATCH_SIZE, THRESHOLD`, so the template raises `SyntaxError: name 'BATCH_SIZE' is used prior to global declaration`. Fix in our script (drop the `global` and pass values through) while keeping the same CLI.
- TensorFlow alternative is included commented out at the bottom (`.h5` or SavedModel; pixel scaling /255).

## Evaluation criteria (100 pts)
| Pts | Criterion | How measured |
|---|---|---|
| 50 | Accuracy on test set | **AUC-ROC** on private labels (`roc_auc_score(true, confidence)`). Score = max(0,(AUC-0.5)/0.5)*50 |
| 20 | Model efficiency | Inference time on T4 for full test set: <5 min = 20, 5-10 = 15, 10-20 = 10, 20-40 = 5, >40 min or OOM = 0 |
| 20 | Generalisation & robustness | Weighted AUC over several undisclosed test-set variants (degradations, quality levels) |
| 10 | Documentation | Manual rubric: problem/approach 2, architecture 2, hyperparameter justification 2, training pipeline 2, limitations 1, reproducibility 1 |

Bonus: Grad-CAM/SHAP/attention maps +2; documented ensemble of diverse architectures +2; export for both PyTorch and TF +1.
Penalties: script does not reproduce predictions -5; OOM/timeout -> efficiency 0.
Disqualification: plagiarism, external training data, hardcoded predictions, failing canary integrity checks.
Tiebreakers: robustness, then efficiency, then documentation, then earlier submission.
Tips from organisers: batched inference, FP16 (`model.half()`), no per-image model loading; make your own 80/20 stratified validation split (`random_state=42`) and track AUC-ROC (also report F1 and threshold choice); the test set mixes **different deepfake generation methods**, so avoid models that only learn one method's surface artefacts or JPEG block artefacts.

## Rules (summary)
Only provided dataset; ImageNet-pretrained allowed; no deepfake-specific pretrained detectors; test images unused for training; one submission per team (latest upload counts); plagiarism = immediate disqualification; no "build inside the event window" rule is stated for this event (dataset was released before the window).

## Dataset facts (verified locally, in `resources/extracted/dataset/hackathon_dataset/`)
- `train/real/` 3,478 JPEGs, `train/fake/` 3,522 JPEGs (7,000 total, near-balanced), 224x224, face-aligned crops.
- `train_metadata.csv`: header `image_id,label` (7,000 rows; image_id = filename without `.jpg`). Label 0 = real, 1 = fake.
- `test/images/`: 3,000 JPEGs, flat, UUID names, no labels.
- `sample_submission.csv` (dataset copy): 3,000 rows, header `image_id,label,confidence`, real test IDs with label 0 / confidence 0.5. (The one at the package top level is a 2-row placeholder.)
- Training data deliberately contains noise, compression artefacts and some annotation ambiguity; robustness to this is expected.
- Audit (3 Nov): in-the-wild face-swap photos; heavy corruptions (snow, block patches, pixelation, cutout, sketch, greyscale, blur, noise) on BOTH classes and test; all JPEGs share one quant table; **real/fake pairs from the same source photo exist** -> grouped CV required. Full plan: `docs/PLAN.md`.

## Resources (`resources/`)
- `Deepfake ML Hackathon - Participant Package-20261103T041932Z-1-001.zip` — original Google Drive download of the participant package.
- `extracted/Deepfake ML Hackathon - Participant Package/README.md` — dataset layout, label convention, submission format, deliverables, inference spec, rules, tips.
- `extracted/.../evaluation_metric.md` — AUC-ROC scoring details, 100-pt breakdown, bonuses, penalties, tiebreakers, validation code.
- `extracted/.../inference_template.py` — official inference script template (PyTorch; TF alternative in comments).
- `extracted/.../sample_submission.csv` — 2-row placeholder showing the CSV format.
- `extracted/.../hackathon_dataset.zip` — nested dataset archive (original).
- `extracted/dataset/hackathon_dataset/` — the dataset unzipped (train/, test/, train_metadata.csv, sample_submission.csv).

## Open questions
- **Metric contradiction:** brief.md says "accuracy"; organiser docs say **AUC-ROC** on the `confidence` column. Optimise for AUC-ROC and calibrated/ranked probabilities.
- **Submission container contradiction:** Unstop text = single zip (report 1-2 pages) vs README = Drive folder (report min 4 pages, source code required). Plan to satisfy both; confirm with the organiser which is final.
- **Report length:** 1-2 pages (Unstop) vs min 4 pages (README); documentation rubric is 10 pts, so follow the 4-page rule.
- **Deadline time:** 15 Nov 11:59 PM vs the Unstop window ending 15 Nov 12:26 AM. Submit by end of 14 Nov.
- **Python version:** README says Python 3.10; template uses pandas (not in the stated framework list but imported) — pin pandas in requirements.txt. `torch` is not installed on this machine; GPU/venv availability unknown.
- Test set mixes unseen generation methods and undisclosed degradation variants; no public leaderboard, so local validation (ideally grouped/robustness-tested) is the only signal.
- Bonus requires both PyTorch and TF exports (+1) — decide whether worth it.
- The Drive link in brief.md (1hs7q-...) is the source of the downloaded package; no other documents were seen. Confirm nothing newer was added.
