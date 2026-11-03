# Deepfake Detection — Master Plan

Planning session: 3 Nov 2026. Safe deadline: **end of 14 Nov 2026**. No code yet.

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
| TF export (+1) | **Skip unless everything else is done**: the PyTorch→ONNX→TF path is fiddly, for 1 point |

---

## 9. Environment

- Local RTX 5060 Laptop (Blackwell, sm_120, 8 GB) needs **PyTorch ≥ 2.7 with CUDA 12.8 wheels**. System Python is 3.12. Create a **Python 3.10** venv (`uv`) to match the brief.
- Kaggle T4 notebook for final timing and a clean-environment reproducibility test.

---

## 10. Timeline (3 → 14 Nov)

| Dates | Milestone |
|---|---|
| **3-4 Nov** | Env setup; (optional) email organisers about CLIP; data tooling: group/pair discovery, grouped folds, pair masks, val-C builder |
| **5-6 Nov** | Baseline: EfficientNet-B0 @224, basic aug → first CV-AUC + val-C. Then the full augmentation stack; ablate each family. |
| **7-8 Nov** | Scale up: EfficientNet-B4@320, ConvNeXt-S, ViT-B. Add EMA, LLRD, aux mask head + pair batching. Compare. |
| **9-10 Nov** | SBI pseudo-fakes experiment; (CLIP only if approved). Pick ensemble members + weights from OOF predictions. |
| **11 Nov** | Final retrains on full data; bundle weights; write inference script; Kaggle T4 timing + double-run determinism check |
| **12-13 Nov** | Grad-CAM figures, loss curves, report (>= 4 pages), README, requirements, packaging (Drive folder + zip) |
| **14 Nov** | Fresh-clone reproducibility test, upload, set Drive to public, submit on Unstop. **Done.** |
| 15 Nov | Buffer only |

Cut order if behind: TF export → SBI → 3rd ensemble member → aux mask head. Never cut: grouped CV, augmentation, inference script tests, report.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Overfitting one generator's artefacts → poor test AUC | Strong corruption aug, SBI, aux mask, diverse ensemble, held-out-cluster check |
| Validation leakage via pairs → overconfident choices | Grouped folds |
| Inference fails on organiser machine (paths, versions, no internet, greyscale input) | No `pretrained=True` at inference, RGB convert, pinned requirements, Kaggle clean-room test |
| Timeout | Measured T4 timing; ensemble kept to ≤3 models |
| Rule ambiguity (CLIP, format) | Email organisers early; default to the safe interpretation |
