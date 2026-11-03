# Phase 2: Robustness and regularisation on B0 (target: 4-5 Nov 2026)

Scope: settle the training recipe (augmentation, regularisation, schedule) cheaply on EfficientNet-B0, and fix *how* we decide things, before spending GPU-hours on big backbones.
Parent plan: `docs/PLAN.md`. Previous phase: `docs/PHASE1.md`. Results go to `docs/EXPERIMENTS.md`.

Guiding rules (unchanged from Phase 1):
- Never touch `test/images/`. Seed 42 for every artefact we keep (exception: the seed-noise diagnostic in step 1, which is labelled as such and never ships).
- **Held-out val-C corruptions (motion blur, shot noise, speckle noise, fog, sketch, posterise) never appear in training**, not even as "similar" library transforms (see the deny-list in 2.3).
- One YAML per experiment; the config *is* the experiment description.

Why B0 for this: at ~4 min per fold (12 min per 3-fold comparison) we can afford ~15 comparisons in a day. A big backbone costs 10-16 min per fold, 3-4x more. Recipe choices like "how much corruption augmentation" and "EMA or not" mostly carry across backbones; architecture-specific knobs (LR, layer decay, drop-path rate) are re-tuned lightly in Phase 3.

---

## 0. Measured costs this plan is built on (3 Nov)

| item | measured |
|---|---|
| B0 train epoch (5.6k imgs, bs 64, BF16), with or without corruption aug | 9-10 s (12 CPU workers keep up) |
| first epoch of a run (cudnn autotune + worker spin-up) | +20-75 s |
| val-C on the best checkpoint (1.4k imgs x 57 settings) | 60-80 s |
| one B0 fold, 12 epochs, incl. val-C | **3.5-4.5 min** |
| 3-fold comparison / 5-fold run | **~12 min / ~20 min** |
| throttling seen in fold 4 of the 5-fold run | epochs went 9 s -> 16 s mid-run (likely thermal / power). Budget x1.3. |

---

## 1. Decision protocol (the most important part of this phase)

### 1.1 What we measure: the *last* (EMA) checkpoint, not the best epoch
Phase 1 picked the best epoch by val AUC. Two problems:
1. **It flatters.** Val AUC jumps around by about ±0.01 from epoch to epoch, so taking the max of 12 noisy numbers adds an upward bias. You saw this: baseline fold 0 "best" was epoch 2, yet epochs 3-12 sit 0.01-0.03 lower.
2. **It cannot be used for the final model.** In Phase 5 we retrain on 100% of the data (see PHASE5 §3), so there is no validation set left to pick an epoch. The recipe has to be good *at a fixed, pre-chosen epoch count*.

So, from Phase 2 on, **the decision checkpoint is the final epoch's weights, preferably the EMA weights** (step 2.5). We still log the best epoch as a diagnostic.

Code: `train.py` gets `select: last | best` (default `last` from now on); val-C runs on the selected checkpoint. `oof_fold{k}.csv` is written from the selected checkpoint.

### 1.2 How many folds, and what difference is real
- **Paired 3-fold comparison** on folds 0, 1, 2. "Paired" means we compare A and B *on the same fold*, then average the three differences. Fold difficulty differs by about ±0.01 (Phase 1), but that variation cancels when both configs see the same fold. This is the same idea as a paired t-test.
- **Metric:** Δ = selection score (0.5 x clean AUC + 0.5 x mean val-C AUC), from the last checkpoint.
- **"Real improvement" rule:** mean Δ ≥ **+0.004** **and** Δ > 0 on all 3 folds.
  - 0.004 is about 4x the cudnn run-to-run noise (±0.001). Step 1 measures the seed noise; if two identical configs with different seeds differ by more than 0.002 per fold, the threshold becomes 2 x that.
  - Requiring all three folds to agree cuts false positives: a pure-noise change passes the sign test with probability 1/8.
- **Ties** (|mean Δ| < 0.004): keep the simpler or cheaper option. If both are equal, prefer the one with higher held-out val-C (held-out is our best proxy for the hidden robustness variants).
- **Veto:** never adopt a change that drops held-out val-C by more than 0.01 or clean AUC by more than 0.01, even if the selection score rises.
- **Confirmation:** the final Phase 2 recipe runs on **all 5 folds** and must beat the B0 baseline OOF (0.8715 at best epoch) by a clear margin at the *last* checkpoint. That 5-fold result is the reference that Phase 3 backbones and Phase 4 tricks must beat.

### 1.3 Greedy, in order
Experiments are run as a greedy forward search: each group starts from the current best recipe and the winner is carried forward. Order matters, so the most impactful and least interacting knobs come first (corruption aug, then EMA/schedule, then the small regularisers).

---

## 2. Code changes (4 Nov morning, ~2-3 h)

| file | function / change | why |
|---|---|---|
| `src/dfd/train.py` | `load_config(path)`: support `base: other.yaml` with recursive dict merge | Configs become true diffs (5-10 lines each). |
| `src/dfd/train.py` | `--fold` accepts `0`, `0,1,2`, `all`; OOF summary for any subset (`oof_f012.csv`, AUC over those folds) | 3-fold protocol. |
| `src/dfd/train.py` | `select: last|best`; val-C + OOF on the selected checkpoint | §1.1 |
| `src/dfd/train.py` | EMA via `timm.utils.ModelEmaV3(model, decay, warmup=True)`; update after each `opt.step()`; validate *both* raw and EMA each epoch (log `val_auc`, `val_auc_ema`); checkpoint stores both | §2.5 |
| `src/dfd/train.py` | optional `seed:` key (default 42) passed to `seed_everything` and `make_loader` | step 1 only |
| `src/dfd/train.py` | `mixup: {alpha, p}` applied on the batch (labels mixed) | §2.7 |
| `src/dfd/train.py` | `accum_steps` (gradient accumulation), unused on B0, needed in Phase 3 | |
| `src/dfd/augment.py` | `SeenCorruptions(p, names, weights, n_max, sev_mode)`: weighted choice of corruption, `sev_mode: discrete | continuous` (continuous samples a severity in [0.5, 3.5] by interpolating parameters) | §2.3 |
| `src/dfd/corruptions.py` | parameterised versions `pixelate_p(img, block)`, `gaussian_noise_p(img, sigma)` etc. used by continuous mode. **The discrete val-C functions stay byte-identical** (val-C must not change, or old results become incomparable). | |
| `src/dfd/augment.py` | `ExtraAug`: extra seen-family transforms (2.3) + `HELD_OUT_DENYLIST` assertion | no leakage |
| `src/dfd/evaluate.py` | save per-image val-C probabilities to `valc_probs.npz` (57 settings x n_val, float16, ~150 kB/fold) | needed for ensemble selection in Phase 5; cheap to add now |
| `scripts/run_queue.py` (new) | runs a list of `(config, folds)` sequentially; **skips runs whose `eval.json` exists** (restartable after a crash or GPU dropout); runs `check_env`-style GPU check before each run and stops the queue if CUDA is gone; logs `nvidia-smi` temp/clock/power every 60 s to `outputs/runs/gpu_monitor.csv` | overnight safety, throttling detection |
| `scripts/compare.py` (new) | `compare.py runA runB [...] --folds 0,1,2` prints the paired-Δ table (clean, seen, held-out, worst, selection; per fold + mean) and the verdict from §1.2 | makes decisions mechanical |

**Acceptance check for the code:** `--limit 512 --epochs 1 --valc_limit 64 --no_log` smoke test passes for (a) the baseline config, (b) a config using `base:` + EMA + mixup + continuous corruptions. Then `baseline_b0` fold 0 with `select: best` and no EMA reproduces the Phase 1 numbers to within ±0.002 (proves nothing broke).

---

## 3. Experiments

All configs live in `configs/p2/`. Reference = `p2_ref.yaml`:
```yaml
base: ../b0_seencorr.yaml
description: "P2 reference: b0_seencorr, last-checkpoint selection"
select: last
```
Each run below = 3 folds (0,1,2) ≈ 12 min unless marked. Times include the x1.3 throttling margin only in the day totals.

### Step 1: reference and noise floor (~25 min)
| run | diff vs ref | purpose |
|---|---|---|
| `p2_ref` | n/a | reference under the new protocol (last checkpoint) |
| `p2_ref_seed43` | `seed: 43` | **Diagnostic only.** Measures what "no real change" looks like across 3 folds. Sets the Δ threshold (§1.2). |

Expect: `p2_ref` last-checkpoint selection score ≈ 0.86 on fold 0 (best-epoch was 0.866; last-epoch clean AUC was 0.881 vs best 0.883, so small).

### Step 2: corruption augmentation strength and mix (~50 min)
| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_corr_p07` | `aug.seen_corruptions_p: 0.7` | More corrupted samples = stronger regulariser and wider coverage. Phase 1 showed +0.03 val-C from p 0 to 0.5. | val-C seen +0.005-0.01, clean ±0.005 |
| `p2_corr_p09` | `aug.seen_corruptions_p: 0.9` | Too much may hurt clean AUC (model rarely sees clean faces). | clean -0.005, val-C ~flat vs p0.7 |
| `p2_corr_weighted` | `aug.corr_weights: {pixelate: 3, downscale: 2, gaussian_noise: 3, snow: 2}` (others 1) | Sample the weak corruptions more often. Pixelate@3 is the worst cell (0.65); noise at sev 3 is next. | pixelate@3 +0.05-0.10; seen +0.005 |
| `p2_corr_cont` | `aug.sev_mode: continuous`, severity range [0.5, 3.5] | Training on severities slightly *beyond* val-C sev 3 makes sev 3 easier. Discrete severities also teach "exactly these 3 levels", which the hidden variants won't match. | worst cell +0.03; held-out flat or + |

Carry forward the best of p ∈ {0.5, 0.7, 0.9}, then test weighting and continuous severity on top of it (both are cheap to stack; run them as one combined run `p2_corr_best_wc` if both individually pass).

### 2.3 Extra augmentation for the weak settings, without leaking held-out types
The three weakest held-out cells are shot noise, speckle noise and motion blur. We may **not** train on them, or on anything that is effectively the same corruption. The point of held-out is to measure how we do on degradations nobody showed the model. So the rule is: **add more variety inside the seen families, and nothing that has a held-out corruption's defining property.**

| allowed extra (seen families) | denied (would leak) | why denied |
|---|---|---|
| Gaussian noise: per-channel *and* luminance-only, σ up to 35 | Poisson / shot noise, `A.ISONoise`, `A.ShotNoise`, any signal-dependent noise | Shot noise = noise that scales with brightness |
| Salt-and-pepper (impulse) noise | Multiplicative / speckle noise, `A.MultiplicativeNoise` | Speckle = x + x·n |
| JPEG q 10-95 (random), double JPEG | | |
| Downscale with random interpolation (nearest / area / linear), pixelate blocks 2-16 | | |
| Gaussian blur σ 0.5-3.5, isotropic defocus disk | `A.MotionBlur`, `A.ZoomBlur`, any directional kernel | Motion blur = directional kernel |
| Snow (ours), `A.RandomRain` (thin streaks; seen "weather" family) | `A.RandomFog`, haze, any low-frequency veil | Fog |
| Gamma, `A.CLAHE`, `A.Equalize`, colour cast, greyscale, hue shift | `A.Posterize`, any colour quantisation / bit-depth reduction | Posterise = fewer intensity levels |
| Coarse dropout / block patches (ours) | `A.Emboss`, edge/sketch filters, `A.Superpixels` | Sketch |

Implementation: `augment.py` defines `HELD_OUT_DENYLIST = {"MotionBlur", "ZoomBlur", "ISONoise", "ShotNoise", "MultiplicativeNoise", "RandomFog", "Posterize", "Emboss", "Superpixels"}` and asserts at construction time that no transform in the pipeline has one of these class names. A unit check in the smoke test builds every config and runs the assertion.

| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_extra` | `aug.extra_p: 0.3` (one random transform from the allowed column, applied before SeenCorruptions) | Wider variety inside seen families generalises to *neighbouring* unseen corruptions (Gaussian noise variety → shot/speckle; blur variety → motion blur). This is the actual test of "robust to the unseen". | held-out +0.005-0.015; seen ~flat |

### Step 3: EMA and schedule (~50 min)
| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_ema` | `ema: {decay: 0.998, warmup: true}`; select EMA weights | EMA = running average of weights over the last few hundred steps. It averages away the epoch-to-epoch spikes, so the *last* EMA checkpoint ≈ the best raw epoch. This is what makes fixed-epoch full-data retraining safe. | last-ckpt selection +0.003-0.01; much smoother `val_auc_ema` curve |
| `p2_ep8` | `train.epochs: 8` | With corruption aug the best epoch was 4 and val loss rises after epoch 3-4. A shorter cosine schedule ends near the sweet spot. | ≈ equal to 12 epochs with EMA, 1/3 cheaper |
| `p2_ep20_lr1e4` | `epochs: 20`, `optim.lr_backbone: 1.0e-4` | Gentler fine-tuning = less destruction of pretrained features, which helps unseen corruptions (the "gentle fine-tuning" lesson). | held-out +; clean ±; 1.7x cost |

Decision note: EMA is expected to win and to change which epoch count is best, so step 3 runs **after** EMA is carried forward (the epoch runs include EMA).

### Step 4: small regularisers (~60 min)
| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_ls05` | `loss.label_smoothing: 0.05` | Organisers state some labels are noisy. Smoothing stops the model being pushed to infinite confidence on mislabelled images (train loss 0.01 = memorised). | AUC ±0.003 (probably a tie); better calibration (lower val loss) |
| `p2_ls10` | `label_smoothing: 0.10` | same, stronger | same |
| `p2_dp02` | `model.drop_path_rate: 0.2` | Stochastic depth randomly skips residual blocks during training: an ensemble-of-subnetworks regulariser. Standard for EfficientNet fine-tuning (timm default 0.2). | +0.002-0.006 |
| `p2_drop05` | `model.drop_rate: 0.5` | More head dropout. Mostly matters for tiny datasets; expected tie. | tie |

### Step 5: mixup / cutmix decision (~12 min)
- **CutMix: no, not run.** CutMix pastes a rectangle of image B into image A and labels the result by area fraction (e.g. 0.3 fake). For deepfakes that label is wrong. A real photo with a fake face pasted in is *fully* fake, and a fake whose face got covered by a real patch may be real. CutMix would teach the model exactly the wrong local logic. SBI (Phase 4) is the principled version of "paste a region and label it fake".
- **Mixup: one test.** Mixup blends two whole images and their labels (0.7·A + 0.3·B). It is not semantically meaningful for real/fake either, but at low strength it's a known calibration and regularisation trick for overfitting models.

| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_mixup` | `mixup: {alpha: 0.2, p: 0.5}` | Softer decision boundary against heavy overfitting | most likely a tie or slightly negative; adopt only by the strict rule |

### Step 6: does upsampling help a CNN? (informs Phase 3, ~25 min)
| run | config diff | hypothesis | expected effect |
|---|---|---|---|
| `p2_res288` | `model.input_size: 288` (bilinear upsample inside `Classifier.forward`, see PHASE3 §2) | Stride-32 CNNs at 224 end on a 7x7 map; 288 gives 9x9, so blending seams get more resolution. Phase 3 plans B4@320 on this premise. | +0.005-0.01 clean; costs ~2x |

If this does not pass the rule, Phase 3 runs B4 at 288 (cheaper) instead of 320.

### Step 7: confirm (~20 min)
`p2_final` (all carried-forward winners) on **5 folds** → OOF AUC, selection score, val-C table. This is the "Phase 2 recipe" and the new baseline.

**Total GPU: ≈ 4 h** (16 three-fold runs ≈ 3.2 h + 5-fold confirm 20 min + margin). Fits one day + one overnight queue.

---

## 4. Go / no-go

- **Go to Phase 3** when `p2_final` 5-fold exists and the code checks pass. Phase 3 does not have to wait for steps 4-6: if late, Phase 3 starts with whatever recipe exists on the morning of 5 Nov.
- **Expected outcome:** last-checkpoint selection score ≈ 0.87-0.88 (vs 0.866 best-epoch for b0_seencorr), worst cell > 0.70.
- **If nothing in steps 3-5 passes the rule:** fine. Ship the simplest recipe; record the ties in the report as "tested, no effect" (hyperparameter justification is 2 documentation points, and negative results count).

**Cut order if late:** step 6 (just use B4@288) → step 5 (mixup: default no) → `p2_drop05`, `p2_ls10` → `p2_ep20_lr1e4`. Never cut: step 1, EMA test, the 5-fold confirmation.

---

## 5. Risks

| risk | mitigation |
|---|---|
| GPU fails to wake from sleep mid-queue (happened 3 Nov) | `power/control=on` (user, sudo) before queues; charger on; `run_queue.py` checks CUDA before each run, stops cleanly, and is restartable (skips finished runs). Check the queue log every morning. |
| Thermal throttling (fold 4 ran 70% slower) | `gpu_monitor.csv`; laptop on a hard surface; budget x1.3. Throttling slows runs but does not change results. |
| Greedy search overfits our 3 folds (many comparisons) | Strict rule (4x noise + 3/3 signs), veto rule, final 5-fold confirmation; Phase 3 runs on all 5 folds anyway. |
| Accidental held-out leakage via a library transform | Deny-list assertion + review of `augment.py` diff before the first run. |
| val-C function change makes old numbers incomparable | Discrete val-C functions frozen; new parameterised functions are separate. Smoke test reruns the baseline (§2 acceptance). |

## 6. Explain to the user at the checkpoints
- Why paired comparison beats comparing raw numbers (fold difficulty cancels).
- What EMA does to the weights (a moving average, like smoothing a noisy loss curve, but done on the parameters).
- Why CutMix is wrong for this task while SBI is right.

## Phase 2 exit criteria
- [ ] Code changes in §2 merged; smoke tests pass; baseline reproduces.
- [ ] Noise floor measured; Δ threshold fixed and written in EXPERIMENTS.md.
- [ ] Every run in §3 logged (or explicitly cut) with a verdict from `compare.py`.
- [ ] `configs/p2/p2_final.yaml` 5-fold OOF + val-C logged.
- [ ] Decisions + one-line reasons appended to a "Decisions" section in EXPERIMENTS.md (feeds the report's hyperparameter table).
