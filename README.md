# Deepfake face detection — Synergy '26 Deepfake ML Hackathon (IIIT Bangalore)

Binary classifier for 224×224 face crops: **real (0)** vs **deepfake (1)**. For every test image it outputs P(fake).

- **Model.** ImageNet-pretrained `timm` backbones (EfficientNet, ConvNeXt, ViT), fine-tuned only on the provided training set.
- **Robustness.** Training uses heavy, label-independent corruption augmentation. Robustness is measured on a 57-setting corruption benchmark (val-C), and 6 of its corruption types are never used in training.
- **Validation.** Grouped 5-fold cross-validation, because real/fake versions of the same photo exist in the data.
- **Final models.** Retrained on 100 % of the data and combined into a single weights bundle.
- **Inference.** Batched FP16 with deterministic output.

The method and every decision behind it are in the technical report (`docs/report/`, built with `scripts/build_report.py`). The run-by-run log is `docs/EXPERIMENTS.md`.

| (grouped 5-fold, out-of-fold) | AUC-ROC | val-C seen | val-C held-out |
|---|---|---|---|
| EfficientNet-B0 baseline | 0.8715 | 0.839 | 0.791 |
| Phase 2 recipe (`configs/p2/p2_final.yaml`) | 0.8964 | 0.883 | 0.871 |
| **final submission** | <!-- TODO(results) --> | | |

---

## 1. Setup

You need Python **3.10** and, for GPU use, an NVIDIA driver with CUDA 12.x support. The pinned torch build is `2.11.0+cu128`, which ships its own CUDA runtime and includes kernels for sm_75 (T4) through sm_120 (RTX 50xx). Without a GPU, everything also runs on CPU, more slowly.

```bash
# with uv (https://docs.astral.sh/uv/); plain `python3.10 -m venv .venv` works too
uv venv --python 3.10 --seed .venv

# (a) inference only: what the organisers need
.venv/bin/pip install -r requirements-infer.txt

# (b) full training environment (every package pinned) + this repo's `dfd` package (editable)
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

Both requirements files carry `--extra-index-url https://download.pytorch.org/whl/cu128`, because the `+cu128` torch wheels live on the PyTorch index. If you use `uv pip install -r ...` instead of pip, add `--index-strategy unsafe-best-match` so uv can take torch from the PyTorch index and everything else from PyPI.

| file | contents |
|---|---|
| `requirements-infer.txt` | torch, torchvision, timm, numpy, pandas, opencv-python-headless, pillow, and their dependencies. Enough for `inference_script.py`. |
| `requirements.txt` | the frozen training venv: the above plus albumentations, scikit-learn, scipy, matplotlib, tensorboard, imagehash, … |

## 2. Data

The dataset is not redistributed in this repo. Unpack the organisers' `hackathon_dataset.zip` anywhere and point `DF_DATA_ROOT` at it:

```
$DF_DATA_ROOT/
├── train/real/*.jpg        3,478 images
├── train/fake/*.jpg        3,522 images
├── train_metadata.csv      image_id,label   (label 0 = real, 1 = fake)
└── test/images/*.jpg       3,000 images (only ever read by inference_script.py)
```

```bash
export DF_DATA_ROOT=/path/to/hackathon_dataset
```

All training code resolves the dataset root from `--data_root` or `$DF_DATA_ROOT`; there are no hardcoded paths. Run outputs go to `outputs/runs/<run>/`, which is git-ignored; set `DFD_RUNS_DIR` to put them elsewhere.

`data/` holds small files derived from **training metadata only**. They are committed, so the steps that create them are optional:

| file | made by | contents |
|---|---|---|
| `data/pairs.csv`, `data/groups.csv`, `data/folds.csv` | `python -m dfd.groups` | near-duplicate pairs → similarity groups → `StratifiedGroupKFold(5, shuffle=True, random_state=42)` |
| `data/masks/`, `data/masks_meta.csv`, `data/mask_prior.npy` | `python -m dfd.masks` | real/fake pair-difference masks (Phase 4) |
| `data/loco_clusters.csv` | `python -m dfd.loco clusters` | fake-image clusters for the leave-one-cluster-out check |

## 3. Reproduce the predictions (what the organisers run)

The organisers' CLI, unchanged:

```bash
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
#   optional: --batch_size 64 (default)   --threshold T (default: the threshold stored in the bundle)
```

- **Input.** The script processes every `.jpg/.jpeg/.png` file in `--test_dir`, case-insensitively and in sorted order, with `image_id` = the file name without its extension.
- **Output.** It writes `image_id,label,confidence`, where `confidence` = P(fake) and `label` = `confidence >= threshold`.
- **Self-contained.** It needs only `requirements-infer.txt` and the single `model_weights.pt` bundle, which holds every ensemble member, its temperature and weight, and the threshold. It does not import the training package and never touches the network: models are built with `pretrained=False` and `HF_HUB_OFFLINE=1` is set.
- **Deterministic.** cuDNN autotuning is off and nothing is random, so two runs on the same machine give a byte-identical CSV.
- **Fast.** Inference is batched, uses FP16 autocast on CUDA, and loads each model once. The CPU fallback runs in FP32.
- **Validate** the output (header, the 3,000 IDs, value ranges, label/threshold consistency):

  ```bash
  python scripts/validate_predictions.py predictions.csv --test_dir ./test/images/
  ```

- **Measured.** The safety-net single-B0 bundle took 13.8 s wall time for the 3,000 test images on an RTX 5060 Laptop GPU.

<!-- TODO(results): T4 wall time of the final bundle; bundle size + SHA-256; Drive link to model_weights.pt. -->

## 4. Train from scratch

All commands run from the repo root with the venv's Python (`.venv/bin/python`, or `python` after activating the venv) and `DF_DATA_ROOT` set. One B0 fold takes about 4 min on an 8 GB laptop GPU. A ConvNeXt-S, ViT-B or EfficientNet-B4@320 fold takes about 9-13 min.

**Configs** are YAML files. A `base:` key inherits from another config and merges recursively, so each experiment is a small diff, and the config itself describes the experiment.

| config | what it is |
|---|---|
| `configs/baseline_b0.yaml` | Phase 1 baseline (basic augmentation) |
| `configs/p2/p2_final.yaml` | **Phase 2 recipe**: B0, corruption aug p 0.9, EMA, drop-path 0.2, label smoothing 0.10, 12 epochs, last EMA checkpoint |
| `configs/p3/{b4_320,cnxs,vitb}_final.yaml` | Phase 3 backbones on the same recipe (EfficientNet-B4@320, ConvNeXt-S, ViT-B/16) |
| `configs/p4/*.yaml` | Phase 4 ablations (SBI pseudo-fakes, mask head, pair batching) |
| `configs/p2/*.yaml` (others), `configs/loco/` | individual ablations and diagnostics; see `docs/EXPERIMENTS.md` |

```bash
# 0. (optional) regenerate groups/folds and pair masks; the committed files are what all results used
python -m dfd.groups candidates                 # hashes + ImageNet embeddings + audit grids (train images only)
python -m dfd.groups folds --threshold 0.82     # threshold confirmed visually on the audit grids
python -m dfd.masks

# 1. cross-validation: one fold, a subset, or all five (OOF predictions + val-C report per fold)
python -m dfd.train --config configs/p2/p2_final.yaml --fold 0
python -m dfd.train --config configs/p2/p2_final.yaml --fold 0,1,2 --skip_done
python -m dfd.train --config configs/p2/p2_final.yaml --fold all
#    -> outputs/runs/p2_final/fold{k}/{last.pt,metrics.csv,eval.json,oof_fold{k}.csv,valc_probs.npz}, oof_all.csv
#    each finished run appends a row to docs/EXPERIMENTS.md (use --no_log to skip)

# 2. paired comparison of two runs on folds 0-2 (decision rule: delta >= 0.006 and positive on all 3 folds)
python scripts/compare.py p2_ema p2_final --folds 0,1,2 --threshold 0.006

# 3. final model: same recipe on 100% of the data, fixed epoch count, last EMA weights
python -m dfd.train --config configs/p2/p2_final.yaml --fold full     # -> outputs/runs/p2_final/full/last.pt

# 4. (optional) flip-TTA out-of-fold predictions for the ensemble search
python -m dfd.evaluate --ckpt outputs/runs/p2_final/fold0/last.pt --tta_flip

# 5. ensemble selection from out-of-fold evidence (temperatures, weights, threshold)
python -m dfd.ensemble --runs cnxs_final b4_320_final vitb_final p2_final [--variant tta_flip]

# 6. one weights bundle for inference_script.py
python scripts/bundle.py --ensemble outputs/ensemble/ensemble.json --source full --out model_weights.pt
python scripts/bundle.py --member p2_final:full --out model_weights.pt            # single model (safety net)
```

**Run queues.** `scripts/run_queue.py` runs several trainings sequentially, which suits overnight runs. It skips finished folds, checks CUDA before each run and enforces a per-run timeout:

```bash
python scripts/run_queue.py queues/p3_final.yaml [--dry_run]
```

Queue entries take `folds:` (`"0"`, `"0,1,2"` or `"all"`). Two caveats:

- The queue runner always launches `.venv/bin/python`, so the venv must live at `.venv`.
- Full-data retrains (`--fold full`) are launched directly with `python -m dfd.train`, not through the queue.

**Smoke test** (about 1 minute, checks that the code path runs end to end):

```bash
python -m dfd.train --config configs/p2/p2_final.yaml --fold 0 --limit 512 --epochs 1 --valc_limit 64 --no_log
DFD_DEVICE=cpu python -m dfd.train ...           # same, on CPU (slow; for tests only)
```

**Other tools.**

| command | purpose |
|---|---|
| `python -m dfd.viz valc` | grid of one validation image under every val-C corruption × severity (`outputs/audit/valC_examples.jpg`) |
| `python -m dfd.evaluate --ckpt <run>/fold0/last.pt` | re-evaluate a checkpoint (clean + val-C) |
| `python -m dfd.gradcam grid --ckpt outputs/runs/<run>/fold0/last.pt` | Grad-CAM grids on **validation** images (`corrupt`, `compare` and `selftest` subcommands also exist) |
| `python -m dfd.loco clusters / configs / report` | leave-one-cluster-out check (unseen-generation-method proxy) |
| `python scripts/plot_curves.py outputs/runs/<run>/fold0 ... --out curves.png` | loss / AUC curves from `metrics.csv` (fold runs) |
| `python scripts/parity_test.py --run p2_final --fold 0 [--device cuda]` | inference script vs training code: logit parity, robustness folder, byte-identical re-run |
| `python scripts/check_env.py` | GPU / torch / BF16 / timm sanity check (expects the RTX 50xx training GPU) |
| `python scripts/kaggle/make_kit.py ...` | T4 timing kit for a Kaggle notebook (see `scripts/kaggle/README.md`) |
| `python scripts/build_report.py` | `docs/report/report.md` → PDF (headless Chrome/Chromium; see the script's docstring) |

## 5. Repository map

```
inference_script.py        self-contained inference (organisers' CLI; template's global-variable SyntaxError fixed)
requirements.txt           full training environment (pinned)      requirements-infer.txt   inference only
pyproject.toml             installs src/dfd as the `dfd` package (Python 3.10)
src/dfd/
  paths.py                 DF_DATA_ROOT / DFD_RUNS_DIR resolution, output dirs
  seed.py                  seeding + deterministic DataLoader workers
  groups.py                near-duplicate pairs -> groups -> grouped folds
  corruptions.py           val-C benchmark: 19 corruptions x 3 severities (13 seen, 6 held out)
  augment.py               training augmentation + held-out deny-list assertion
  data.py                  datasets, loaders, pair masks, group batch sampler
  models.py                timm wrapper (in-model normalisation/upsampling, zero-init head, aux mask head, LLRD)
  train.py                 training loop (folds, --fold full, EMA, label smoothing, OOF, val-C)
  evaluate.py              clean + val-C evaluation, flip TTA, pair/non-pair breakdown
  ensemble.py              temperature scaling, ensemble search, threshold
  masks.py, sbi.py         pair-difference masks, self-blended pseudo-fakes (Phase 4)
  loco.py                  leave-one-cluster-out diagnostic
  gradcam.py, viz.py       Grad-CAM and audit grids
configs/                   one YAML per experiment (`base:` inheritance)
queues/                    run_queue.py queue files
scripts/                   run_queue, compare, bundle, validate_predictions, parity_test, plot_curves,
                           build_report, check_env, kaggle/ (T4 timing kit)
data/                      folds, groups, pairs, masks (derived from train metadata only)
docs/                      PLAN.md, PHASE1-6.md (plans), EXPERIMENTS.md (every run + decision), report/
```

## 6. Rules compliance (short)

- **Training data.** Only the provided training images are used. Pseudo-fakes and masks are derived from them.
- **Pretrained weights.** Only ImageNet-1k/21k/22k weights (`timm` tags `*_in1k`, `*in21k_ft_in1k`, `*in22k_ft_in1k`). No CLIP/DINO-style weights and no deepfake-specific detectors.
- **Test images.** They are read only by `inference_script.py`, never for training, selection or calibration.
- **Fixed seeds.** Seed 42 is used throughout. The full experiment history is in `docs/EXPERIMENTS.md`.

Our code only; the dataset belongs to the organisers and is not included.
