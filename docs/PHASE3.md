# Phase 3: Bigger, diverse backbones (target: 5-7 Nov 2026)

Scope: train 3-4 architecture families with the Phase 2 recipe on all 5 folds, so Phase 5 can pick a diverse ensemble from out-of-fold (OOF) predictions.
Parent plan: `docs/PLAN.md` §4. Previous: `docs/PHASE2.md`.

Why several families (link to what you know): a CNN looks at the image through small sliding windows and builds up texture detectors, so it's good at local blending seams and skin-texture mismatch. A ViT splits the image into 16x16 patches and lets every patch attend to every other one from the first layer, so it's better at global consistency (lighting, face vs. body, identity cues). They make *different* mistakes, so averaging them cancels some of each. That is also the "+2 diverse ensemble" bonus.

Allowed weights only: timm tags trained on **ImageNet-1k or ImageNet-21k/22k**. Do **not** use `*.dinov3_*`, `*.eupe_*`, `*.dino`, `*.mae`, `*.sam_*`, CLIP/EVA/SigLIP tags. ImageNet-12k (`in12k`) tags are a subset of ImageNet-21k and therefore allowed, but we don't need them: each slot below has an in1k/in22k tag.

---

## 1. Candidates and measured cost (probe, 3 Nov; random weights, BF16 train, FP16 infer)

| slot | timm tag | input | train batch | GPU s/epoch | peak VRAM | FP16 infer img/s (local) | est. min/fold* | est. 5-fold* |
|---|---|---|---|---|---|---|---|---|
| (ref) | `efficientnet_b0.ra_in1k` | 224 | 64 | 8.6 | 2.8 GB | 3260 | 4 | 20 min |
| CNN-texture | `efficientnet_b4.ra2_in1k` | 320 (upsampled) | **16** | 43 | 3.8 GB | 471 | 15 | **1.6 h** |
| CNN-modern | `convnext_small.fb_in22k_ft_in1k` | 224 | 48 (64 peaks at 6.7 GB, too tight) | ~29 | ~5.3 GB | 660 | 10 | **1.0 h** |
| Transformer | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 224 | 64 | 34 | 5.5 GB | 587 | 11 | **1.1 h** |
| Transformer alt | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | 224 | 32 | 38 | 4.2 GB | 462 | 12 | 1.2 h |
| cheap alt | `convnext_tiny.fb_in22k_ft_in1k` | 224 | 64 | 18 | 4.3 GB | 1101 | 5 | 0.5 h |

\* 12 epochs + ~1 min first-epoch overhead + val-C (80k forward passes at the inference rate, ~2-3 min) + 30% throttling margin. If Phase 2 settles on 8 epochs, subtract ~1/3 of the training part.

**Probe finding: EfficientNet-B4 at 288/320 and EfficientNetV2-S at 288 *hang* (>120 s, no progress) at batch 32.** Batch 16 is fine. The likely cause is cudnn autotuning (`cudnn.benchmark=True`) searching workspace-hungry algorithms when memory is nearly full. Step 0 tests this; the fallback is batch 16, which is fine (BatchNorm statistics at 16 are OK for fine-tuning).

---

## 2. Code changes (5 Nov, ~2 h)

| file | change | why |
|---|---|---|
| `src/dfd/models.py` → `Classifier` | `input_size` arg; in `forward`, if `input_size != 224`: `F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)` after normalisation | Upsampling on the GPU, inside the model, means data loading, val-C and the inference script stay at 224 and cannot drift apart. |
| `src/dfd/models.py` → `Classifier.__init__` | zero-init generalised: works for timm heads that are `nn.Linear` under `head.fc` (ConvNeXt) or `head` (ViT/Swin); assert that the 1-output head was found and zeroed | Lesson 1 from Phase 1, for every family |
| `src/dfd/models.py` → `param_groups` | if `optim.layer_decay` is set: use `timm.optim.param_groups_layer_decay(model.net, weight_decay, layer_decay=ld, no_weight_decay_list=model.net.no_weight_decay())`, then set each group's `lr = lr_backbone * lr_scale`; the head group gets `lr_head` | Layer-wise LR decay (LLRD) |
| `src/dfd/train.py` | `accum_steps` (already added in Phase 2); `eval_batch_size` key (default 128) independent of train batch | B4 trains at 16 but should evaluate at 128 |
| `src/dfd/train.py` | `grad_ckpt: true` → `model.net.set_grad_checkpointing(True)` | Only if a batch doesn't fit; trades ~25% speed for ~40% activation memory |
| `src/dfd/train.py` | read `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from the queue runner env | reduces fragmentation OOMs near 8 GB |
| `scripts/run_queue.py` | per-run `timeout_min` (kill a run that hangs like the B4 probe, log it, continue) | overnight safety |

**LLRD in one paragraph:** a pretrained network's early layers detect generic edges and textures that are already good; the late layers are task-specific. LLRD gives layer *k* of *L* the learning rate `lr x decay^(L-k)`, so with decay 0.75 on 12 ViT blocks, the first block learns at 0.75^12 ≈ 3% of the top block's LR. Fine-tuning is "gentle" at the bottom, which preserves the robust generic features. That's the main reason fine-tuned ViTs generalise to unseen corruptions.

---

## 3. Steps and runs

All configs in `configs/p3/`, each `base: ../p2/p2_final.yaml` plus the diff shown.

### Step 0: fit checks and weight downloads (~20 min, 5 Nov)
- Download the pretrained weights for every candidate once (needs internet; they cache under `~/.cache/huggingface`). Inference never downloads anything (it uses `pretrained=False` + our bundle).
- Probe B4@320 at batch 24 and 32 with `cudnn.benchmark=False` (timeout 120 s each). If batch 32 works without benchmark, use `deterministic: true` for B4 (it also gives bit-reproducible training, a bonus for the report). Otherwise batch 16.
- 1-epoch smoke run of each config with `--limit 512`.

### Step 1: per-backbone LR sanity on fold 0 (~1.5 h, 5 Nov afternoon)
New architectures can silently fail with the wrong LR (e.g. ViT at 3e-4 often collapses to predicting 0.5). We run a 2-point LR check on fold 0 for each, then commit to 5 folds. This is coarse (single fold, ±0.01 noise), but it only needs to catch big failures (Δ > 0.02).

| run | config diff (vs p2_final) | hypothesis |
|---|---|---|
| `b4_320_lr2e4` | `model: {name: efficientnet_b4.ra2_in1k, input_size: 320, drop_path_rate: 0.2}`, `train.batch_size: 16`, `optim.lr_backbone: 2.0e-4` | Batch is 4x smaller than B0, so LR is scaled by about √(16/64) from 3e-4 → 1.5-2e-4 |
| `b4_320_lr1e4` | same, `lr_backbone: 1.0e-4` | |
| `cnxs_lr1e4` | `model: {name: convnext_small.fb_in22k_ft_in1k, drop_path_rate: 0.2}`, `batch_size: 48`, `lr_backbone: 1.0e-4`, `layer_decay: 0.8` | ConvNeXt fine-tuning recipes use ~1e-4 with LLRD 0.7-0.8 |
| `cnxs_lr5e5` | same, `lr_backbone: 5.0e-5` | |

**Implemented deviation (3 Nov):** timm's fine-grained LLRD grouping gives ConvNeXt-S a 37-step span (not ~12), so `layer_decay: 0.8` would leave the stem at 2.6e-4 x lr (nearly frozen). ConvNeXt-S configs use **0.93 = 0.8^(12/37)** (same stem/top ratio as 0.8 over 12 groups). Equivalents if needed: ConvNeXt-T 0.875, B4 0.924, Swin-S 0.902; ViT-B stays 0.75. Also `lr_lambdas` now decays every LLRD group to the same fraction (min_lr/lr_backbone) of its own peak.
| `vitb_lr5e5` | `model: {name: vit_base_patch16_224.augreg2_in21k_ft_in1k, drop_path_rate: 0.1}`, `batch_size: 64`, `lr_backbone: 5.0e-5`, `layer_decay: 0.75`, `grad_clip: 1.0` | Standard ViT-B fine-tune (BEiT/MAE recipes) |
| `vitb_lr1e4` | same, `lr_backbone: 1.0e-4` | |

`lr_head` stays at 1e-3 (the head starts from zero and needs to move fast). Pick the better LR per backbone by fold-0 selection score; if they're within 0.01, take the **lower** LR (gentler is safer for robustness).

### Step 2: 5-fold runs (overnight 5→6 Nov, ~4 h incl. margin)
Queue (`run_queue.py`, ordered by expected value, so the most important finishes first if the night is cut short):
1. `cnxs_final` 5 folds (~1.0 h): ConvNeXt is the most likely strong single model under corruption.
2. `b4_320_final` 5 folds (~1.6 h)
3. `vitb_final` 5 folds (~1.1 h)

### Step 3: alternates (6 Nov, daytime, ~1.5 h, only if needed)
- If ViT-B is clearly worse than both CNNs (selection < CNN − 0.02): try `swin_s` 5 folds (~1.2 h). Swin is a transformer with local windows, closer to CNNs, and often fine-tunes more easily on small datasets.
- If B4 is close to ConvNeXt-S *and* slow: try `cnxt_tiny` (~0.5 h) as a cheaper second CNN.
- If time remains: `b4_288` (if Phase 2 step 6 was skipped or ambiguous).

### Step 4: Phase 3 report (CPU, ~30 min)
`scripts/compare.py` across all 5-fold runs plus:
- per-model OOF AUC, val-C seen/held-out/worst, selection score;
- **pairwise OOF logit correlation** between models (lower = more diverse = more ensemble gain);
- a quick equal-weight average of the top-3 OOF logits, as a preview of Phase 5.

---

## 4. Expected results and go/no-go

| model | expected 5-fold selection (B0 P2 recipe ≈ 0.87-0.88) |
|---|---|
| B4@320 | +0.01-0.02 |
| ConvNeXt-S in22k | +0.01-0.03 (in22k pretraining = more robust features) |
| ViT-B in21k + LLRD | ±0.02 (highest variance: depends on LR) |

- **Go to Phase 4** with the **best one or two** backbones (Phase 4 experiments re-run on B0 first, then port winners to these).
- A backbone **enters the ensemble pool** only if its 5-fold selection score ≥ (best single) − 0.02 **or** its OOF correlation with the best is < 0.8 *and* the equal-weight pair average beats the best single model.
- **No-go case:** if no big model beats the B0 P2 recipe by ≥ 0.01 on 5 folds, something is off (LR, head init, normalisation). Debug for at most half a day (check that the zero-init head assert holds, look at the curves, try the other LR); otherwise fall back to an ensemble of B0 + B2/B4.

**Cut order if late:** step 3 alternates → ViT-B (keep 2 CNNs: still "diverse" if one is EfficientNet and one ConvNeXt, but the bonus case is weaker) → B4 LR check (go straight to 2e-4).

---

## 5. Risks

| risk | mitigation |
|---|---|
| 8 GB VRAM: OOM or the B4 batch-32 hang | Measured batch sizes above, ~1.5 GB headroom each; `expandable_segments`; per-run timeout; `grad_ckpt` switch. Don't run anything else on the GPU during queues (browser hardware acceleration also uses VRAM). |
| GPU dropout or sleep overnight | `power/control=on`, charger, restartable queue; at worst we lose one fold (≤ 15 min) |
| T4 time budget | Inference rates above are local; a T4 is roughly 2-3x slower in FP16. Three members ≈ 16 s locally → ~50 s on a T4, plus flip-TTA x2 → ~100 s: under the 5-min limit (PHASE5 §5). Five fold-models per member would be ~4-8 min: **too slow**, which is one reason for full-data retraining (PHASE5 §3). |
| ViT fine-tuning instability | grad clip 1.0, warmup 1 epoch, LLRD, zero-init head, the LR check |
| Bigger models overfit even faster (7k images) | EMA + last checkpoint + Phase 2 corruption aug; 8-12 epochs |

## Phase 3 exit criteria
- [ ] Every chosen backbone has 5-fold OOF predictions (`oof_all.csv`) and `valc_probs.npz` per fold, from the last (EMA) checkpoint.
- [ ] EXPERIMENTS.md has one row per fold + OOF row per backbone, and the LR decisions.
- [ ] Comparison table with OOF correlations written to `docs/EXPERIMENTS.md` (section "Phase 3 summary").
- [ ] Decision recorded: which 1-2 backbones carry Phase 4 tricks, which ≤ 4 enter the Phase 5 pool.
