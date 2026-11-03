# Deepfake Detection — Master Plan

Planning session: 3 Nov 2026. Safe deadline: **end of 14 Nov 2026**.
Status (3 Nov evening): Phase 1 done (`docs/PHASE1.md`). Phases 2-6 planned in detail in `docs/PHASE2.md` … `docs/PHASE6.md`; the day-by-day calendar is in §10.

---

## 1. What the data actually is (audited 3 Nov)

| Finding | Evidence | Consequence |
|---|---|---|
| **In-the-wild photos**, loose face crops, often multiple people, backgrounds, low resolution | Contact sheets of real/fake/test | Not FFHQ-style aligned portraits. Context matters; the manipulated region is only part of the image. |
| **Fakes are face swaps blended into real photos** (not fully synthetic faces) | Visual inspection | The signal is local: blending boundary, texture/colour mismatch, identity inconsistency. Classic Face X-ray / SBI territory. |
| **Heavy corruptions on BOTH classes and test**: snow, coloured block patches, pixelation, grey cutout squares, sketch/edge filter, greyscale, blur, noise, colour casts | Visual inspection | Corruptions are **not** a label cue. Hidden "robustness variants" will likely be similar or stronger. Our augmentation must reproduce this family. Closely resembles the OpenForensics "Test-Challenge" corruption set. |
| **All JPEGs re-encoded at one identical quality** (same quant tables, 4:2:0) in train and test | Quant-table check: 1 distinct table across all 10k files | No JPEG-quality shortcut exists. Good. Still augment JPEG because hidden variants may recompress. |
| Test file sizes slightly smaller at p90 (17 kB vs 24 kB) | File-size stats | Test may be a bit blurrier or more degraded than train. More reason for robustness focus. |
| **Real/fake pairs from the same source photo exist** | Perceptual hash: 37 real-fake pairs at Hamming <= 10 bits, 689 at <= 25; visually confirmed same scene with swapped face | (a) Random splits leak, so we need **grouped CV**. (b) Pairs give **free localisation supervision** (difference = manipulated region). (c) Pair-aware batches force the model to ignore background/scene. |
| Label noise / ambiguity stated by organisers | README | Label smoothing, robust loss; no aggressive data cleaning. |

> We must never use test images for training (even unsupervised). Test images are only for final prediction (and the visual sanity look above).

---

## 2. Rules → technology boundaries

| Allowed | Gray area (ask organiser before using) | Forbidden |
|---|---|---|
| ImageNet-1k / ImageNet-21k pretrained backbones (EfficientNet, ConvNeXt, ViT, Swin, ResNet...) via `timm` | **CLIP / DINOv2 / EVA weights**: not ImageNet-trained (web-scale data). Best published generalisation uses CLIP, but the rule text says "ImageNet architectures". | External datasets (FF++, Celeb-DF, OpenForensics...) |
| Any augmentation of provided images, incl. **synthesising fakes from provided reals** (SBI-style) | Off-the-shelf face landmark detectors (for SBI masks) | Deepfake-specific pretrained detectors/weights |
| Ideas from papers (not weights) | | Any use of test images in training |

**Decisions (3 Nov). Local college hackathon, so we follow the most reasonable reading of each rule and do not block on organiser replies:**
- ImageNet-1k and ImageNet-21k weights: allowed. They are the whole core system.
- General face-landmark detectors: allowed in principle (not deepfake-specific), but avoided if pair-derived or elliptical masks suffice.
- CLIP/DINOv2: excluded from the submission by default (letter of the rule says "ImageNet"; disqualification risk outweighs a few AUC points). We may run it as an experiment; if the gain is large, decide explicitly and disclose it in the report.
- Submission: satisfy both versions (public Drive folder with everything + zip inside + link on Unstop).
- Report: >= 4 pages, with a 1-page executive summary up front.
- Deadline: end of 14 Nov.
- An email to organisers is optional (mainly to ask about CLIP); nothing in the plan waits on it.

---

## 3. Inspiration from the literature (ideas only)

| Work | Core idea | What we take |
|---|---|---|
| **Face X-ray** (Li et al., CVPR 2020) | Fakes contain a blending boundary; predict the boundary map | Auxiliary localisation head supervised by pair-difference masks |
| **SBI: Self-Blended Images** (Shiohara & Yamasaki, CVPR 2022) | Make pseudo-fakes by blending a real face with a slightly transformed copy of itself; train real vs pseudo-fake; generalises across unseen methods | Generate extra fakes from our **real** images, using masks learned from the pairs. Directly targets "unseen generation methods". |
| **DFDC winner** (Seferbekov 2020) | EfficientNet-B7, heavy augmentation, face-part cutout | EfficientNet family, strong augmentation, cutout of face parts |
| **UCF / LSDA / DeepfakeBench (2023-24)** | Generalisation fails when models overfit method-specific artefacts; disentangle or enlarge the forgery space | Motivation for SBI and for an architecture-diverse ensemble |
| **UnivFD, Effort (ICML 2025)** | Frozen or minimally adapted CLIP ViT features generalise far better than fully fine-tuned CNNs | Gray-area option (section 2). The ImageNet analogue: ViT with layer-wise LR decay (gentle fine-tuning). |
| **NPR / FreqNet (CVPR 2024)** | Upsampling and frequency artefacts | Mostly not used: our data's uniform JPEG + corruptions destroy high-frequency traces; relying on them hurts robustness. Kept only implicitly via CNN branch. |
| **OpenForensics** (Le et al., ICCV 2021) | In-the-wild multi-face swaps + challenge corruptions | Tells us what the hidden robustness variants probably look like (snow, fog, block distortion, pixelation, edge filters, noise, blur, colour). |

---

## 4. Architecture

### 4.1 Overall design: a diverse 2-3 model ensemble

```
                 ┌─────────────────────────────────────────┐
image 224x224 ──►│ Model A: EfficientNet-B4 (CNN, local     │──► logit_A
                 │   texture & blending), input upsampled   │
                 │   to 320 + aux mask head                 │
                 ├─────────────────────────────────────────┤
              ──►│ Model B: ConvNeXt-Small (IN-22k)         │──► logit_B
                 │   modern CNN, strong under corruption    │
                 ├─────────────────────────────────────────┤
              ──►│ Model C: ViT-B/16 or Swin-S (IN-21k)     │──► logit_C
                 │   global context, identity/lighting      │
                 └─────────────────────────────────────────┘
                     weighted average of calibrated logits ──► sigmoid ──► P(fake)
```

Why diversity: CNNs key on local texture and blending seams; transformers on global consistency. Different failure modes give a robust average and the "+2 diverse ensemble" bonus. Final membership is decided by grouped-CV AUC on clean **and** corrupted validation (section 6), not by taste.

### 4.2 Each model internally

- **Backbone:** `timm` pretrained weights (ImageNet only).
- **Classification head:** global pool → dropout(0.3) → Linear(→1) → logit.
- **Auxiliary mask head (Models A/B):** small decoder on a mid-resolution feature map (e.g. stride 16) → 1-channel manipulation map. Supervised by pair-difference masks (or SBI masks) where available, ignored elsewhere. Loss weight ~0.1-0.3. Dropped at inference (no speed cost). Purpose: forces attention onto the face region, not background/scene cues.
- **Input resolution:** native 224 for ViT/ConvNeXt; try upsampling to 288-320 for EfficientNet (CNNs with stride-32 lose fine detail at 224). Keep only if CV-AUC improves.

### 4.3 Candidate shortlist (pick by experiment)

| Slot | First choice | Alternates | Params | Rough T4 FP16 time for 3k images |
|---|---|---|---|---|
| CNN-texture | EfficientNet-B4 @ 320 | EfficientNet-B2, EfficientNetV2-S | 19M | ~20-30 s |
| CNN-modern | ConvNeXt-Small in22k | ConvNeXt-Tiny in22k | 50M | ~15-25 s |
| Transformer | ViT-B/16 augreg in21k | Swin-S in22k, DeiT-III-S | 86M | ~20-30 s |

Total, including image decoding and flip TTA, should stay well under the **5-min** full-points limit. Verify on a Kaggle T4 before submitting.

---

## 5. Training recipe

### 5.1 Data and splits
1. **Pair/group discovery:** cluster near-duplicates (perceptual hash, then confirm with ImageNet-feature cosine similarity, since corruptions break pure hashes). Each cluster is one group.
2. **5-fold StratifiedGroupKFold** (seed 42) on those groups. Fold 0 also serves as the organiser-suggested "80/20 stratified, random_state=42" split we report.
3. **Pair masks:** for confident real-fake pairs, compute |fake − real| after light blur and alignment, threshold, morphological cleanup → soft mask of the swapped region. Skip pairs whose corruptions differ too much (noisy masks).

### 5.2 Augmentation (the robustness engine)
Applied independently of label, probability-tuned so ~50-70% of samples get at least one corruption:
- **Geometric:** horizontal flip, small shift/scale/rotate (±10%, ±10°). No heavy random crops that cut out the face.
- **Photometric:** brightness/contrast/saturation/hue jitter, greyscale, colour cast, gamma.
- **Degradation (mirror observed + likely hidden):** JPEG recompression q30-95, downscale-upscale (×0.25-0.9), Gaussian/motion/defocus blur, Gaussian/shot/ISO noise, pixelation.
- **Occlusion/overlay:** snow/rain/fog, coarse dropout (grey squares), coloured block patches, sketch/edge filter, random face-part cutout (DFDC-winner trick).
- **SBI pseudo-fakes (phase 3):** take a real image, warp/colour-shift a copy, blend it back through a face-region mask (from pair masks or an elliptical mask at the face). Labelled fake. Mixed into batches at ~20-30%.
- **Pair-aware batching:** when a pair exists, put both images in the same batch so the model must find the difference, not the scene.

### 5.3 Optimisation
| Hyperparameter | Value | Why |
|---|---|---|
| Loss | BCE with label smoothing 0.05 (+0.2 × mask BCE/Dice) | Label noise is stated; smoothing also improves calibration. |
| Optimiser | AdamW, wd 0.05 | Standard for fine-tuning ConvNeXt/ViT. |
| LR | Head 1e-3; backbone 1e-4 (CNN) / 5e-5 with layer-wise decay 0.75 (ViT) | Preserve pretrained features; layer-wise decay protects early layers (the "gentle fine-tuning" lesson from CLIP work). |
| Schedule | 1-epoch linear warmup, cosine decay | Stable start, smooth convergence. |
| Epochs | 15-25, best checkpoint by CV-AUC | 7k images; augmentation slows overfitting. |
| Batch | 32-64 with BF16 autocast; gradient accumulation if VRAM-limited | Fits the 8 GB RTX 5060. |
| Regularisation | Dropout 0.3, stochastic depth 0.1-0.2, EMA of weights | EMA weights are consistently more robust. |
| Seeds | Fixed (torch/numpy/random), deterministic dataloader seeding | Reproducibility requirement. |

### 5.4 Final models
After choosing configs by 5-fold CV, either (a) retrain each chosen architecture on 100% of the training data with the CV-chosen epoch count, or (b) keep the fold models. Prefer (a) for speed. If time allows, compare against (b) using fold-out-of-fold predictions.

**Decided (3 Nov, PHASE5 §3):** (a). Five fold models x 3 members would take an estimated 4-8 min on a T4 (efficiency risk). To make (a) safe, all decisions from Phase 2 on use the *last EMA checkpoint at a fixed epoch count*, not the best epoch. Fallback for a member whose full-data run looks abnormal: average its fold-0 and fold-1 models.

---

## 6. Evaluation: our only leaderboard

- **Primary:** out-of-fold AUC-ROC (grouped 5-fold). Also report F1, accuracy, and the chosen threshold on fold 0.
- **Robustness suite "val-C":** each validation image × {JPEG q20/40, blur σ1/2/3, noise σ10/25, downscale ×0.5/0.25, snow, fog, block patches, pixelation, greyscale, sketch, brightness ±, contrast ±} × severities → mean and worst-case AUC. This approximates the hidden 20-pt robustness score.
- **Selection rule:** score = 0.5 × clean AUC + 0.5 × mean val-C AUC, mirroring 50 + 20 points' worth of AUC.
- **Unseen-method proxy:** cluster fake images by ImageNet features into generation-style clusters. Train with one cluster held out and test on it. A rough check that we aren't learning one method's fingerprint. Optional.
- **Grad-CAM sanity check:** heatmaps should land on the face/boundary, not backgrounds or corruption overlays.

---

## 7. Inference (efficiency: 20 pts, reproducibility: −5 risk)

- Same CLI as the template: `--test_dir --weights --output [--batch_size --threshold]`. Fix the template's `global` SyntaxError.
- `--weights` points to **one file** bundling all ensemble members' state_dicts + config (architecture names, input sizes, ensemble weights, temperatures, threshold). No hardcoded paths; architectures built with `pretrained=False` (no internet needed at grading).
- Robust I/O: `.convert("RGB")` (greyscale/RGBA variants), resize any size to 224, handle jpg/jpeg/png (+ uppercase).
- Batched, FP16, `torch.inference_mode()`, `cudnn.benchmark`, DataLoader with workers. CPU fallback works (just slower).
- TTA: horizontal flip only (doubles cost; keep only if it helps val-C).
- Ensemble: average temperature-scaled logits with CV-tuned weights → sigmoid.
- `label` = confidence ≥ threshold (threshold chosen on validation for F1; AUC unaffected).
- Deterministic: same weights → bit-identical CSV (to 6 decimals) on re-run. We verify by running twice and diffing.
- Time it on a **Kaggle T4** end-to-end, including model loading.

---

## 8. Deliverables

| Item | Plan |
|---|---|
| predictions.csv | 3,000 rows, validated (IDs match test dir, no NaN, range [0,1]) |
| model_weights.pt | Single bundled checkpoint, < 1 GB |
| inference_script.py | Section 7 |
| requirements.txt | Python 3.10, torch/torchvision (pinned), timm, pandas, numpy, pillow |
| Source code | `src/` (data, augment, models, train, eval, gradcam) + README with exact repro commands; GitHub + zip |
| Report (>= 4 pages PDF) | Problem, data audit (pairs, corruptions), architecture, hyperparameter justification table, training pipeline, CV + val-C results, ablations, Grad-CAM, limitations, reproducibility |
| Logs/curves | CSV/TensorBoard exports → plots in report |
| Grad-CAM (+2) | Grid of correct/incorrect real/fake with heatmaps |
| TF export (+1) | **Committed (user decision, 3 Nov)**: PyTorch→ONNX→onnx2tf→SavedModel in a separate `.venv-tf`, on 13 Nov, hard stop 18:00 (PHASE6 §4) |
| Source code hosting | **Public GitHub repo** (user decision, 3 Nov), with no Claude co-author lines; old trailers scrubbed before the first push, only with the user's OK at that time (PHASE6 §5) |

---

## 9. Environment

- Local RTX 5060 Laptop (Blackwell, sm_120, 8 GB) needs **PyTorch ≥ 2.7 with CUDA 12.8 wheels**. System Python is 3.12. Create a **Python 3.10** venv (`uv`) to match the brief.
- Kaggle T4 notebook for final timing and a clean-environment reproducibility test.

---

## 10. Timeline (revised 3 Nov evening; detail in PHASE2-6)

Phase 1 (env, grouped folds, val-C, B0 baseline) finished on 3 Nov, about 2 days ahead of the original plan.
Budget assumption (user, 3 Nov): **daytime runs (~4-6 h) + overnight queues (~6-8 h)**, so roughly 12 GPU-h/day of capacity. Overnight queues need `power/control=on` (set by the user with sudo), the charger plugged in, the restartable `run_queue.py`, and a morning check.
GPU-h = local RTX 5060 busy time incl. a 30% throttling margin. Run-time basis: B0 ≈ 4 min/fold; B4@320 ≈ 15, ConvNeXt-S ≈ 10, ViT-B ≈ 11 min/fold (probe-based, PHASE3 §1).

| date | phase | daytime (attended) | overnight queue | GPU-h |
|---|---|---|---|---|
| **Wed 4 Nov** | 2 | Code: config `base:`, multi-fold, last/EMA select, queue runner, compare.py, aug extensions + held-out deny-list, valc_probs. Smoke tests. Runs: noise floor, corruption p/mix, extra aug. **You:** Kaggle account + phone verification. | P2 steps 3-6 (EMA, epochs, LS, drop-path, mixup, res288) | ~4 |
| **Thu 5 Nov** | 2→3 | Pick the P2 recipe → 5-fold confirm. Phase 3 code (input upsample, LLRD, eval batch, timeouts). Weight downloads, B4 batch probe, LR sanity on fold 0 for B4 / ConvNeXt-S / ViT-B. | 5-fold: ConvNeXt-S → B4@320 → ViT-B | ~6 |
| **Fri 6 Nov** | 3→4 | Phase 3 summary (+ Swin-S / ConvNeXt-T alternate if needed). Phase 4 code: pair masks, **mask audit with you**, SBI generator, **SBI audit with you**, mask head, pair batching, LOCO clusters. | B0 Phase 4 ablations (3-fold) + LOCO | ~3.5 |
| **Sat 7 Nov** | 4 | Analyse Phase 4 (pair/non-pair breakdown, LOCO, Grad-CAM spot check); decide adoptions. | Port winners to 1-2 big backbones, 5 folds | ~4 |
| **Sun 8 Nov** | buffer / 5 | Reruns of anything that failed overnight. Start Phase 5 code: `ensemble.py`, `bundle.py`, `--fold full`. | (spare: 3rd backbone with Phase 4 recipe if needed) | ~2 |
| **Mon 9 Nov** | 5 | Flip-TTA OOF runs; ensemble selection (members, temperatures, weights, threshold). **Feature freeze in the evening.** | Full-data retrains of all members | ~1.5 |
| **Tue 10 Nov** | 5 | `inference_script.py`, parity test, validator, double-run, offline run. **Kaggle T4 timing + sm_75 check.** | — | ~0.5 |
| **Wed 11 Nov** | 5→6 | Fixes; **final predictions.csv v1 on a T4; upload v1 to Drive.** Grad-CAM grids, curves, val-C chart, ablation table. | — | ~0.5 |
| **Thu 12 Nov** | 6 | Report §1, §3, §4; README; requirements-infer; **history scrub (your OK) + public GitHub push.** | — | 0 |
| **Fri 13 Nov** | 6 | **TF export 09:00-18:00 (hard stop)**; report §2, §5-7, summary; build the PDF. Unstop window is open (from 12 Nov 23:59). | — | ~0.2 |
| **Sat 14 Nov** | 6 | Package; Drive public check (incognito); fresh-clone + Kaggle re-run; **submit on Unstop by ~18:00**; tag `v1.0-submission`. | — | ~0.3 |
| Sun 15 Nov | — | Buffer only. Round window reportedly closes 00:26. | — | — |
| **total** | | | | **~22-23 GPU-h** of ~120 available: it fits with a large margin; the real constraint is your attention for the audits and decisions |

Milestones that must not slip: P2 recipe (5 Nov AM), 5-fold big backbones (6 Nov AM), feature freeze (9 Nov PM), **valid predictions.csv on Drive (11 Nov)**, submission (14 Nov).

Cut order if behind (first cut first): Phase 2 minor ablations (mixup, drop-rate, LS 0.10, res288) → Phase 3 alternates (Swin-S / ConvNeXt-T) → pair batching → LOCO → porting Phase 4 to a 2nd backbone → mask head → tuned ensemble weights (use equal) → flip TTA → 3rd ensemble member → SBI. TF export is committed but time-boxed (13 Nov, 18:00 stop). Never cut: grouped CV, corruption augmentation, last-checkpoint protocol, inference script tests (double-run, validator, T4 timing), report, public Drive check.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Overfitting one generator's artefacts → poor test AUC | Strong corruption aug, SBI, aux mask, diverse ensemble, held-out-cluster check |
| Validation leakage via pairs → overconfident choices | Grouped folds |
| Inference fails on organiser machine (paths, versions, no internet, greyscale input) | No `pretrained=True` at inference, RGB convert, pinned requirements, Kaggle clean-room test |
| Timeout | Measured T4 timing; ensemble kept to ≤3 models |
| GPU fails to wake from sleep / drops out overnight (seen 3 Nov) | `power/control=on`, charger, restartable `run_queue.py` with a CUDA check before each run, morning check; critical runs scheduled with a day of slack |
| 8 GB VRAM (B4 at batch 32 hangs at 288-320; ConvNeXt-S at 64 peaks at 6.7 GB) | Probe-measured batch sizes (PHASE3 §1), `expandable_segments`, per-run timeout, grad checkpointing switch |
| Thermal throttling (an epoch went 9 s → 16 s on 3 Nov) | `gpu_monitor.csv`; x1.3 time margin in every estimate |
| Rule ambiguity (CLIP, format) | Email organisers early; default to the safe interpretation |
