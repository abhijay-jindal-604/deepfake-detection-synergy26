# Phase 1 — Foundations (target: 3-6 Nov 2026)

Scope: environment, grouped folds, robustness benchmark, first baseline.
Parent plan: `docs/PLAN.md`. Project facts and rules: `CLAUDE.md`.

Guiding rules for this phase:
- **Never touch `test/images/` for anything except final inference.** No embeddings, hashes or statistics on test images in this phase.
- No hardcoded absolute paths. The dataset root is a CLI argument (default read from env var `DF_DATA_ROOT`).
- Everything seeded (`42`) and re-runnable from a single command per step.
- The user is a CS student learning as we go. Explain *why* at each checkpoint, briefly.

---

## 0. Repo skeleton

```
05-deepfake-ml/
├── CLAUDE.md, brief.md
├── docs/            PLAN.md, PHASE1.md, EXPERIMENTS.md (results log)
├── configs/         baseline_b0.yaml (later: one yaml per experiment)
├── src/dfd/         python package
│   ├── paths.py         dataset root resolution (CLI/env), output dirs
│   ├── seed.py          seeding + deterministic dataloader workers
│   ├── groups.py        pair/duplicate discovery -> groups, folds
│   ├── corruptions.py   deterministic corruption suite (val-C)
│   ├── augment.py       training augmentation pipelines
│   ├── data.py          Dataset classes, transforms, loaders
│   ├── models.py        timm model factory (+ later: aux mask head)
│   ├── train.py         training loop (CLI: --config, --fold)
│   ├── evaluate.py      clean AUC/F1 + val-C report for a checkpoint
│   └── viz.py           contact sheets / pair grids for audits
├── data/            generated metadata (groups.csv, folds.csv, pairs.csv). Small, committed.
├── outputs/         runs/, audit/ (git-ignored)
├── resources/       organiser files (git-ignored: large)
├── requirements.txt (training env)   requirements-infer.txt (later, minimal)
└── .gitignore
```

- `git init` on a fresh `main`, with `.gitignore` covering `resources/`, `outputs/`, `.venv/`, `*.pt`. First commit = skeleton + docs.

---

## 1. Environment

**Goal:** Python 3.10 venv where PyTorch sees the RTX 5060 (Blackwell, compute capability 12.0).

Steps:
1. Check `uv` is available (install via the official installer if not, after asking the user).
2. `uv venv --python 3.10 .venv`.
3. Install PyTorch **CUDA 12.8** wheels (`--index-url https://download.pytorch.org/whl/cu128`): torch >= 2.7 + matching torchvision. Older/cu12.1 wheels do not support sm_120.
4. Install: `timm`, `albumentations` (pin the exact version; the 1.x→2.x API changed), `opencv-python-headless`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `pyyaml`, `tensorboard`.
5. Freeze exact versions into `requirements.txt`.

**Acceptance check (run and show the output):**
- `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available() == True`, device name, `get_device_capability() == (12, 0)`.
- A BF16 autocast matmul on GPU runs.
- `timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)` downloads and does a forward pass on a (2,3,224,224) GPU tensor → shape (2,1).

**Explain to the user:** what a venv is for, why CUDA wheel versions must match the GPU generation, what BF16 is.

---

## 2. Pair / group discovery + grouped folds

**Goal:** every training image gets a `group_id` so that all images derived from the same source photo (real, its fake, and differently-corrupted copies) share a group. Folds split by group.

### 2a. Similarity signals (train images only)
1. **Perceptual hashes:** 64-bit pHash (DCT) and dHash on grayscale. Cheap, but corruptions (snow, patches) break them.
2. **Deep embeddings:** ImageNet-pretrained `convnext_tiny` (or `resnet50`), global-pooled, L2-normalised. Compute on a lightly denoised copy (median blur 3) to reduce sensitivity to snow/noise. 7k images, so this takes about 1 minute on the GPU.
3. For each image, take the top-10 nearest neighbours by cosine similarity (7k × 7k matrix fits in memory) plus all pairs with pHash Hamming <= 10.

### 2b. Threshold calibration (**human-in-the-loop checkpoint**)
- Bucket candidate pairs by cosine similarity (e.g. 0.95+, 0.90-0.95, 0.85-0.90, 0.80-0.85).
- Render a grid of ~16 random pairs per bucket to `outputs/audit/pairs_<bucket>.jpg` (left/right with labels and score).
- Look at the grids yourself, propose a threshold where pairs stop being "same source photo", **show the grids to the user and confirm** before proceeding.

### 2c. Groups and folds
- Union-find over accepted pairs → connected components = groups. Guard against giant components (if one group > ~50 images, the threshold is too loose; tighten).
- Outputs:
  - `data/pairs.csv`: `id_a,id_b,label_a,label_b,cos_sim,phash_dist`
  - `data/groups.csv`: `image_id,label,group_id,group_size`
  - `data/folds.csv`: `image_id,label,group_id,fold` via `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- Report: #pairs by type (real-fake / real-real / fake-fake), #groups, group-size histogram, per-fold count and fake ratio (each fold should be ~20% of the data and ~50% fake).
- Fold 0 is the organiser-style "80/20 stratified split, random_state=42" used in the report.

### 2d. (Stretch, not required this phase) pair masks
For real-fake pairs with cos >= the strict threshold: |fake − real| on blurred grayscale → threshold → morphological open/close → largest components → save `data/masks/<fake_id>.png`. Visual-audit 16. Used later by the aux mask head / SBI. Skip if time is short.

**Explain to the user:** why random splits leak with paired data (the model "recognises the photo"), what union-find does, what StratifiedGroupKFold guarantees.

---

## 3. Robustness benchmark "val-C"

**Goal:** a deterministic function that, given validation images, produces corrupted copies over a fixed list of corruption × severity, plus an evaluator that reports AUC on each.

### 3a. Corruption list (3 severities each; deterministic per (image_id, corruption, severity) seed)
| Family | Corruptions | Seen in training aug? |
|---|---|---|
| Compression | JPEG q {60, 35, 15} | seen |
| Resolution | downscale→upscale ×{0.6, 0.4, 0.25}; pixelate {block 4, 8, 12} | seen |
| Blur | gaussian σ {1, 2, 3} | seen |
| | **motion blur** k {7, 11, 15} | **held out** |
| Noise | gaussian σ {8, 16, 28}; | seen |
| | **shot/poisson + speckle** | **held out** |
| Weather/overlay | snow (density levels) | seen |
| | **fog** | **held out** |
| Occlusion | grey cutout squares {1, 3, 6}; coloured block patches {5, 15, 30} | seen |
| Colour | greyscale; brightness ±{0.2, 0.35, 0.5}; contrast ×{0.7, 0.5, 0.3}; saturation/colour cast | seen |
| Filters | **sketch/edge filter**, **posterise** | **held out** |

"Held-out" corruptions are never used in training augmentation. They measure generalisation to *unseen* degradations (the organisers' hidden variants are unknown).

### 3b. Implementation notes
- Pure functions `corrupt(img_uint8_rgb, name, severity, seed) -> img_uint8_rgb` in `corruptions.py`; albumentations/OpenCV under the hood, custom code for block patches, snow, sketch, posterise.
- Output always 224×224 RGB uint8, so it plugs straight into the normal eval transform.
- Generate on the fly in `evaluate.py` (no disk cache needed; 1.4k val images × ~45 settings ≈ 63k forward passes ≈ 1-2 min for B0).
- `outputs/audit/valC_examples.jpg`: one image × every corruption × severity, for a visual check that each corruption looks right and severity increases.

### 3c. Report format (`evaluate.py` output → `outputs/runs/<run>/eval.json` + printed table)
- clean AUC, F1@0.5, F1@best-threshold, best threshold, accuracy
- per corruption: AUC at each severity
- summary: mean val-C AUC (seen), mean val-C AUC (held-out), worst-case AUC
- **selection score = 0.5 × clean AUC + 0.5 × mean val-C AUC (all)**

**Explain to the user:** why AUC is threshold-free, why held-out corruptions matter, how this proxies the 20-pt robustness score.

---

## 4. Baseline: EfficientNet-B0

**Goal:** the first honest numbers, and a working train/eval pipeline that later experiments reuse by changing only a YAML file.

### 4a. Config (`configs/baseline_b0.yaml`)
| Key | Value |
|---|---|
| model | `efficientnet_b0` (timm, ImageNet-1k pretrained), `num_classes=1`, drop_rate 0.3 |
| input | 224, ImageNet mean/std |
| train aug | hflip 0.5, ShiftScaleRotate(0.06, 0.1, 10°) p 0.5, ColorJitter(0.2,0.2,0.2,0.02) p 0.5. Basic only, by design. |
| loss | BCEWithLogits, label smoothing 0.0 (baseline) |
| optimiser | AdamW, lr backbone 3e-4, head 1e-3, wd 0.05 |
| schedule | 1-epoch linear warmup → cosine to 1e-6 |
| epochs / batch | 12 / 64, BF16 autocast, channels_last |
| seed | 42, deterministic worker seeding |
| checkpoint | best by val AUC + last; save config + metrics alongside |

### 4b. Training loop requirements
- `python -m dfd.train --config configs/baseline_b0.yaml --fold 0 --data_root <path>`
- Per epoch log: train loss, val loss, val AUC, val F1, lr, epoch time → `metrics.csv` + TensorBoard.
- At the end: run `evaluate.py` (clean + val-C) on the best checkpoint.
- Save out-of-fold predictions `oof_fold{k}.csv` (`image_id,label,prob`) for later ensembling.
- `--folds all` runs 5 folds sequentially and writes the OOF AUC across all 7k images.

### 4c. Runs in this phase
1. Fold 0 baseline (smoke test → full). Read the curves together with the user: is it over- or underfitting?
2. If time allows: all 5 folds → OOF AUC (variance across folds tells us how noisy our validation is).
3. Stretch: same config + full "seen" corruption augmentation (from `augment.py`) on fold 0, to show its effect on val-C. This is the first ablation for the report.

### 4d. Results log
Append every run to `docs/EXPERIMENTS.md`:
`date | run name | model | key changes | clean AUC | val-C seen | val-C held-out | worst | selection score | notes`

**Explain to the user:** what each logged curve means, how to read train/val divergence, what a good next experiment would be.

---

## Phase 1 exit criteria
- [ ] Env verified on GPU; `requirements.txt` frozen.
- [ ] `data/folds.csv` exists; pair thresholds confirmed visually with the user.
- [ ] `val-C` examples grid looks right; evaluator prints the full table.
- [ ] Baseline B0 fold-0 results in `docs/EXPERIMENTS.md` (ideally 5-fold OOF too).
- [ ] Git repo initialised, work committed.

Next (Phase 2, ~7-10 Nov): full augmentation stack, bigger backbones (EfficientNet-B4@320, ConvNeXt-S, ViT-B), EMA, LLRD, label smoothing, aux mask head + pair batching, SBI, ensemble selection.
