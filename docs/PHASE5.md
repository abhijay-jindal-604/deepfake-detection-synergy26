# Phase 5: Ensemble, final models, inference (target: 9-11 Nov 2026)

Scope: choose the ensemble from out-of-fold evidence, train the final models, build the single weights bundle and `inference_script.py`, and prove the script is fast on a T4 and reproducible. **By the end of 11 Nov a complete, valid `predictions.csv` exists.** Everything after that is documentation and polish.

Parent plan: `docs/PLAN.md` §6-7. Previous: `docs/PHASE4.md`.

**Feature freeze: evening of 9 Nov.** After that, no new training ideas; only final retrains and fixes.

---

## 1. Inputs
For every candidate member (≤ 4 in the pool; Phase 3 §4 decides), from its 5-fold run at the last (EMA) checkpoint:
- `oof_all.csv` (7,000 OOF probabilities),
- `fold{k}/valc_probs.npz` (per-image val-C probabilities, 57 settings, added in Phase 2).

## 2. Choosing members and weights (`src/dfd/ensemble.py`, CPU, ~2 h coding; runs in seconds)

1. **Logits.** Convert each member's OOF probability to a logit `z = log(p/(1-p))`.
2. **Temperature per member.** Fit one scalar `T_m` per member by minimising OOF log-loss of `sigmoid(z/T_m)` (`scipy.optimize.minimize_scalar`). Why: models trained with/without label smoothing or SBI have very different logit scales (a ViT may output ±3 where a CNN outputs ±12). Averaging raw logits would let the most over-confident model dominate. Dividing by T puts them on the same "calibrated" scale. (A per-model bias would not change the ensemble ranking, so AUC only depends on T and the weights.)
3. **Objective** J = 0.5 x OOF AUC + 0.5 x mean val-C AUC of the combined score `s = Σ w_m z_m / T_m`, computed per fold and setting from `valc_probs.npz` (the same selection score as everywhere else, now for the ensemble).
4. **Search:** all subsets of the pool of size 1-4 x weights on a simplex grid (step 0.05). That's ~2k combinations, under a minute.
5. **Anti-overfitting rules:**
   - **Equal weights are the default.** Use tuned weights only if J beats equal weights by ≥ 0.003.
   - **Optimism check:** choose weights on folds 0-2, evaluate on folds 3-4; report both. If the tuned weights lose on 3-4, use equal weights.
   - A member stays only if removing it lowers J by ≥ 0.002 (each member costs inference time).
6. **Diversity bonus check (+2):** the final ensemble should contain ≥ 2 *architecture families* (CNN + transformer, or EfficientNet + ConvNeXt at minimum). If the best J uses one family only and a 2-family ensemble is within 0.003, take the 2-family one: the bonus is worth more than 0.003 AUC (0.003 AUC = 0.3 + ~0.1 points; the bonus = 2 points).
7. **Threshold** for the `label` column: maximise F1 on the ensemble's OOF probabilities. Report F1, accuracy and the threshold (the organisers' tip). AUC doesn't depend on it.
8. **Final ensemble temperature** `T_ens` on the OOF combined score, so `confidence = sigmoid(s / T_ens)` is calibrated (a 0.8 means ~80% fake). Not needed for AUC, but good practice, and it's in the report.

Output: `outputs/ensemble/ensemble.json` = {members, T_m, w_m, T_ens, threshold, OOF/val-C metrics, fold-0 F1/acc}, plus a table for the report.

## 3. Final models: retrain on 100% of the data (decision)

Options: (a) retrain each member once on all 7,000 images with the CV-chosen epoch count; (b) ship the 5 fold models per member.

**Choice: (a), with (b) as a fallback.** Reasons:
- **T4 budget.** 3 members x 5 folds = 15 forward passes per image. Estimated ~4-8 min on a T4, which falls into the 15-point band or worse. With (a), 3 passes ≈ 1-1.5 min (§5).
- **More data:** +25% training images, and each pair's both sides are seen together.
- **Safety:** this is why Phase 2 switched to the *last EMA checkpoint at a fixed epoch count*. The recipe is validated as "train N epochs, take the end" rather than "pick the lucky epoch".
- The cost of (a): the final models have no validation set of their own. Mitigation: their training curves must look like the fold runs (train loss within ±10% at each epoch). If one looks abnormal, use (b-lite) for that member: average its **fold-0 and fold-1** models (2 passes instead of 1).

Code: `train.py --fold full` trains on all 7,000 images; no per-epoch validation (it would be on training data); saves `final.pt` (EMA weights); skips val-C.
Runs (overnight 9→10 Nov): one per member, seed 42. Estimated B4@320 ~15 min, ConvNeXt-S ~9 min, ViT-B ~10 min (one fold's training x 1.25 for the extra data, without val-C), so **~35-45 min total**.

Same epoch count as the folds: the dataset is 25% larger, so it's 25% more steps, but the same number of passes over the data. That's the standard choice.

## 4. Weights bundle and inference script

### 4.1 `scripts/bundle.py` → `model_weights.pt` (single file)
```python
{
  "format_version": 1,
  "members": [
     {"timm_name": "convnext_small.fb_in22k_ft_in1k", "input_size": 224,
      "drop_rate": 0.0, "state_dict": {...fp16...}, "temperature": T, "weight": w},
     ...
  ],
  "temperature_ens": T_ens, "threshold": thr, "tta_flip": bool,
  "mean": [...], "std": [...], "train_git_commit": "<sha>", "created": "<date>"
}
```
- `mask_head.*` keys are dropped; weights are stored in FP16 (halves the size). Expected size: ConvNeXt-S ~100 MB + ViT-B ~172 MB + B4 ~36 MB ≈ **~310 MB** (limit 5 GB).
- `bundle.py` re-loads the bundle, rebuilds each model with `pretrained=False`, and asserts `load_state_dict(strict=True)`.

### 4.2 `inference_script.py` (repo root, single self-contained file)
- **Exact template CLI:** `--test_dir` (required), `--weights` (required), `--output` (default `predictions.csv`), `--batch_size` (default 64), `--threshold` (default: from the bundle; the CLI overrides). The template's `global BATCH_SIZE, THRESHOLD` SyntaxError is fixed by passing values as arguments (no globals).
- **Self-contained:** imports only torch, timm, numpy, pandas, cv2 (all pinned). It does *not* import `dfd`: the organisers get one script + one weights file. The `Classifier` normalisation and upsampling are copied into the script (~30 lines), and `tests/test_parity.py` checks the script's model gives the same logits as `dfd.models` on 64 val images (max abs diff < 1e-4 in FP32).
- **I/O robustness:** glob `*.jpg *.jpeg *.png` (+ uppercase variants), sorted; `image_id = path.stem`; `cv2.imread(IMREAD_COLOR)` (same decoder as training; turns greyscale into 3 channels, drops alpha); resize to 224 with `INTER_AREA` if needed (same as `data.read_rgb`); unreadable file → fail loudly with the filename (a silent default would be a "hardcoded prediction").
- **Speed:** a single model load; DataLoader with `num_workers=min(4, os.cpu_count())`; batched; CUDA → FP16 (`model.half()`, the organisers' tip), `torch.inference_mode()`, `channels_last`; CPU fallback in FP32.
- **Determinism:** `cudnn.benchmark=False`, `cudnn.deterministic=True`, fixed sorted order, no randomness anywhere; confidence written with `float_format="%.6f"`.
- **No network:** `timm.create_model(name, pretrained=False, num_classes=1)`. Test it with networking disabled (`HF_HUB_OFFLINE=1` plus Kaggle "internet off").
- Same output checks as the template: row count = number of images, unique IDs, no NaN, confidence in [0,1], label ∈ {0,1}.

### 4.3 Flip TTA: decided by data
Run each member's 5 fold models with and without horizontal-flip averaging on OOF + val-C (inference only, ~10-15 min per member). **Adopt flip TTA only if ensemble J improves by ≥ 0.003 *and* the T4 time with TTA stays < 3 min** (keeping 2 min of margin under the 5-minute line). Prior: training already uses hflip p=0.5, so the gain is usually small (+0.001-0.004).

### 4.4 `scripts/validate_predictions.py`
`--pred predictions.csv --test_dir <dir> [--threshold t]` checks: exact header `image_id,label,confidence`; the set of IDs equals the stems in `test_dir` (3,000), no duplicates, no extras; label int ∈ {0,1}; confidence float in [0,1], no NaN; `label == (confidence >= threshold)`. It also prints the class balance and how many confidences lie within ±0.001 of the threshold (labels that could flip on a re-run on different hardware).

## 5. T4 time budget (efficiency: 20 points for < 5 min)

| component | local RTX 5060 (measured / derived) | T4 estimate (2-3x slower in FP16) |
|---|---|---|
| Python + torch + timm import, model build, bundle load (~310 MB) | ~5-10 s | 10-20 s |
| JPEG decode of 3,000 x ~20 kB (4 workers) | ~3-5 s | 5-15 s (CPU-dependent) |
| ConvNeXt-S fwd (660 img/s) | 4.5 s | 9-14 s |
| ViT-B/16 fwd (587 img/s) | 5.1 s | 10-15 s |
| EfficientNet-B4 @320 fwd (471 img/s) | 6.4 s | 13-19 s |
| **total, no TTA** | **~25-35 s** | **~50-90 s** |
| total with flip TTA | ~40-50 s | ~1.5-2.5 min |
| 5 fold models per member instead of 1 (no TTA) | ~1.5-2 min | **~4-8 min (too risky)** |

**Measure, don't trust:** Kaggle T4 notebook (account set-up on 4 Nov), Python 3.10 venv created with `uv` inside the notebook, `pip install -r requirements-infer.txt`, then `time python inference_script.py ...` **on 3,000 *training* images** for timing (so test images aren't needed for that). Also check `nvidia-smi` shows a T4 and that the torch 2.11 cu128 wheel supports sm_75 (it should; if not, pin the last cu12x build that does in `requirements-infer.txt`, re-check parity).

Budget rule: target **≤ 2.5 min** measured end-to-end. If it's over: drop TTA → drop the slowest member (usually B4@320, or run it at 288) → batch 128.

## 6. Reproducibility checks (the −5 penalty)
1. **Double run, same machine:** run the script twice → `cmp predictions_run1.csv predictions_run2.csv` must be byte-identical.
2. **Submitted CSV generated on a T4** (Kaggle, test images in a *private* Kaggle dataset, or Colab reading from your Drive). The organisers re-run on a T4, and FP16 results differ slightly between GPU generations (Blackwell vs Turing), so producing our CSV on the same GPU class minimises mismatches. Run it twice there as well.
3. **Cross-device sanity:** local GPU vs T4 vs CPU: report max |Δconfidence| (expected ≲ 1e-3) and the number of label flips (expected 0-2, all near the threshold). This goes in the report's reproducibility section.
4. **Offline test:** run with networking off.

## 7. Steps and calendar

| when | step | GPU |
|---|---|---|
| 9 Nov AM | flip-TTA OOF runs for pool members | ~45 min |
| 9 Nov AM | `ensemble.py`: T, weights, threshold; decision write-up | CPU |
| 9 Nov PM | `train.py --fold full`; `bundle.py`; start `inference_script.py`; **feature freeze** | — |
| 9→10 Nov night | full-data retrains of all members | ~45 min (+ margin) |
| 10 Nov | inference script, parity test, validator, double-run; Kaggle T4 timing + sm_75 check | ~0.5 h |
| 11 Nov | fixes; final `predictions.csv` v1 generated on the T4; validator passes; **upload v1 to the Drive folder** (a safety copy: any later upload replaces it) | ~0.3 h |

## 8. Go / no-go
- **Must-have by 11 Nov evening:** valid `predictions.csv` + bundle + script that passes double-run, validator, parity, and T4 time ≤ 2.5 min.
- **If behind on 10 Nov:** ship the best *single* backbone (full-data retrain) with no TTA; the ensemble comes back only if it's ready by 12 Nov midday.

**Cut order:** tuned weights (use equal) → TTA → 3rd member → full-data retrain (use fold-0 + fold-1 models).

## 9. Risks
| risk | mitigation |
|---|---|
| Full-data model silently worse | Curve-match check; fold-model fallback (§3) |
| Ensemble weights over-fit OOF | Equal weights by default; folds 0-2 → 3-4 optimism check |
| T4 slower than estimated | Measured on Kaggle by 10 Nov; cut order above |
| Torch wheel lacks sm_75 | Checked on Kaggle on 10 Nov; fallback pin |
| Organiser re-run mismatch (−5) | Deterministic script; CSV produced on a T4; near-threshold count reported |
| Inference box has no internet / different paths | `pretrained=False`, everything from CLI args, offline test |
| GPU dropout during the full retrains | Small (~45 min), restartable queue; done by 10 Nov morning, so a full day of slack |

## Phase 5 exit criteria
- [ ] `ensemble.json` with members, temperatures, weights, threshold, OOF + val-C metrics, optimism check.
- [ ] `model_weights.pt` bundle (< 1 GB) loads strictly; parity test passes.
- [ ] `inference_script.py` keeps the template CLI; double-run byte-identical; offline run OK; validator passes.
- [ ] Kaggle T4 end-to-end time measured and logged (≤ 2.5 min target).
- [ ] `predictions.csv` v1 produced on the T4 and uploaded to Drive.

## Implementation notes (code written 4 Nov, ahead of schedule)
Code: `train.py --fold full`, `src/dfd/ensemble.py`, `src/dfd/evaluate.py --tta_flip`, `scripts/bundle.py`, `inference_script.py`, `scripts/parity_test.py`. `$DFD_RUNS_DIR` overrides `outputs/runs` everywhere.
- **`--fold full`** writes `<run>/full/last.pt` (EMA weights in `state_dict_ema` when EMA is on; bundle.py picks them), `metrics.csv` (train-loss curve for the §3 curve-match check) and `done.json` (the `--skip_done` marker). `select: best` is forced to last. Not `final.pt` as §3 said.
- **TTA (§4.3):** `python -m dfd.evaluate --ckpt <run>/fold{k}/last.pt --tta_flip` writes `fold{k}/tta_flip/{oof_fold{k}.csv,valc_probs.npz,eval.json}`; `python -m dfd.ensemble --variant tta_flip` reads those; bundle.py sets `tta_flip` from the ensemble.json variant (or `--tta_flip`). Compare the two ensemble.json J values for the ≥ 0.003 rule.
- **Ensemble details:** J uses pooled OOF AUC + mean val-C AUC over (fold x setting). If any run lacks `valc_probs.npz`, J is OOF-only for all candidates (warning). Optimism split: folds 0-2 / 3-4 with 5 folds; otherwise all-but-last / last. Weight grid has every weight ≥ step (zero weight = a smaller subset).
- **Deviations from §4.1-4.2:** (1) bundle weights stored **FP32 by default** (`--dtype fp16` available): FP16-rounding the weights alone moved B0 logits by up to 0.09; the size is still far under the limit (~620 MB for the planned 3 members). (2) CUDA path uses `torch.autocast(float16)` instead of `model.half()` (keeps norms/softmax in FP32), with a per-batch FP32 recompute if any logit is non-finite. (3) cudnn.benchmark off + deterministic on (re-run identity beats the small speed gain). Decoder is OpenCV like training; PIL only as a fallback for files OpenCV cannot read.
- **Parity finding:** the training-time OOF/val-C probabilities are computed under **BF16 autocast**, which moves B0 logits by up to ~0.75 vs FP32 (per-image |dp| up to ~0.19, mean ~0.0065 on fold 0) while AUC changes by only ~7e-4. So the script cannot match `oof_fold{k}.csv` to 1e-3 per image; parity is checked exactly against `dfd.models` in FP32 (max |dp| 5.6e-7) and by AUC against the OOF file.
- **Deviation from §4.2 "unreadable file -> fail loudly" (coordinator decision, 4 Nov):** a file that neither OpenCV nor PIL can decode now gets a row with confidence 0.5 (label per threshold), a `[WARNING]` naming it, and a count in the final summary. Reason: a crash scores zero on the organisers' re-run and a missing row scores as wrong, while one neutral 0.5 costs at most that image; it's a documented fallback, not a hardcoded prediction. Also: if DataLoader workers die (e.g. small `/dev/shm` in a container: "worker ... killed" / "shared memory"), the whole pass is redone with `num_workers=0`; same inputs and batches, so the CSV is identical (checked by simulating the failure).
