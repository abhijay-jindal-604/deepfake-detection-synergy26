# Phase 4: Use the real/fake pairs, and self-blended pseudo-fakes (target: 6-8 Nov 2026)

Scope: three ideas that attack the organisers' main warning ("the test set mixes **different deepfake generation methods**"), plus checks that we are not just memorising quirks of our few pairs.
1. **Pair-difference masks** → an **auxiliary mask head** that learns *where* the manipulation is.
2. **Pair-aware batching**: real and fake versions of the same photo in one batch.
3. **SBI pseudo-fakes**: fakes synthesised from our own real images, so the model learns the *generic* signature of face blending instead of one generator's fingerprint.

Parent plan: `docs/PLAN.md` §3, §4.2, §5.2. Previous: `docs/PHASE3.md`.
Everything is developed and ablated on **B0 (3 folds, ~12 min per run)**, then the winners are ported to the 1-2 best Phase 3 backbones.

Rule check: everything here is built from **training images only** (real images → SBI; train pairs → masks). Test images are untouched. No face detector is used by default (see §3.2 for the one gray-area option, which needs your OK).

---

## 1. Pair-difference masks (`src/dfd/masks.py`, CPU, ~2 h coding, 6 Nov)

What we have: `data/pairs.csv` holds 187 real-fake pairs (98 real→fake + 89 fake→real orderings), plus same-label pairs we don't use here. A real-fake pair is the same source photo with the face swapped, so `|fake − real|` is large on the swapped face and small elsewhere: a free "where is the fake" label.

Steps (`build_pair_masks(pairs, root) -> data/masks/<fake_id>.png + data/masks_meta.csv`):
1. Read both images; convert to grey, Gaussian blur σ=2 (suppresses noise and snow speckles).
2. **Align**: `cv2.phaseCorrelate` for a translation; reject the pair if the shift is > 8 px or the response is < 0.3 (different crop).
3. `d = |fake − real|`, blurred again; threshold by Otsu; morphological open (5x5) then close (11x11); keep the largest connected component; fill holes; feather the edge (blur 7) → soft mask in [0,1].
4. **Quality filters** (logged per pair in `masks_meta.csv`): reject if mask area < 2% or > 45% of the image (too small = noise; too big = the pair differs by corruption, not by face swap), or if the two images received visibly different corruptions (mean |d| outside the mask > a threshold calibrated on the audit grid).
5. If one fake has several real partners, keep the mask from the partner with the highest cosine similarity.

**Human-in-the-loop checkpoint (like the pair threshold in Phase 1):** render `outputs/audit/pair_masks.jpg` (32 random accepted + 16 rejected, as real | fake | mask overlay). You and I look at it together and adjust thresholds before any training uses the masks.

Expected yield: roughly 100-170 fake masks (2-5% of fakes). That's thin on its own. But **every real image also has a valid mask: all zeros** ("nothing manipulated here"), and **every SBI pseudo-fake has an exact mask** (§3). Masks only become plentiful in combination with SBI, which is why `p4_sbi_mask` is the run we expect to win.

Also computed here: `data/mask_prior.npy`, the mean of all accepted masks (where swapped faces usually are in these crops). SBI uses it.

`data/masks/` is small (PNG, ~1-2 MB in total), derived from train data only, and committed.

**Implemented deviations (4 Nov 00:00):** translation-only phaseCorrelate yielded 1 usable mask (most pairs are re-crops zoomed 0.8-1.2x, shifted up to ~45 px). Now: affine `findTransformECC` alignment (cc>=0.90, geometry/coverage gates), quality gate cc>=0.97, constant-colour block exclusion, robust brightness/contrast match, `MAX_OUTSIDE_D=8.5`. Yield: **31 accepted / 187 pairs** (~18 clean by eye), far below the 100-170 expected; the mask head will rely mainly on SBI masks + all-zero real masks. Many `pairs.csv` real-fake 'pairs' (cos_sim < 0.6, pHash <= 10) are different people: groups are over-merged (harmless for CV), and §6.2 defines 'pair' fakes by successful ECC alignment instead.

---

## 2. Auxiliary mask head (`src/dfd/models.py`, ~2 h)

**Idea (Face X-ray):** besides "fake or not", ask the network to draw a map of the manipulated pixels. It can't do that by keying on background, scene or corruption cues, so the shared backbone is pushed to represent the face region and its boundary. That helps exactly when the generation method changes.

Implementation:
- `Classifier(..., mask_head=True)`: take a **stride-16 feature map** via timm's `forward_intermediates` (14x14 at 224, 20x20 for B4@320; for ViT, the last block's patch tokens reshaped to 14x14).
- Head: `Conv3x3(C→64) → BN → ReLU → Conv1x1(64→1)`, about 0.1-0.4 M params.
- `forward(x)` returns `logit` (and `mask_logits` in train mode).
- Loss: `L = BCE(logit, y) + λ · BCE(mask_logits, mask_14x14) [only where mask_valid]`, λ = 0.2 by default. Masks are area-downsampled to the feature size.
- `mask_valid`: real images → 1 (all-zero target), fakes with a pair mask → 1, SBI fakes → 1, other fakes → 0 (ignored).
- Data: `FaceDataset` returns `(img, label, mask, mask_valid)`; geometric aug is applied to image and mask together (albumentations `Compose(..., additional_targets)`). Photometric/corruption aug touches the image only.
- **Inference:** the mask head is not called (`mask_head` keys are dropped when the weights are bundled; PHASE5 §4). Zero inference cost.

---

## 3. SBI pseudo-fakes (`src/dfd/sbi.py`, ~3 h)

**Idea (Self-Blended Images, CVPR 2022):** every face-swap pipeline ends the same way: paste a generated face region into a target photo and blend the edge. That leaves generic traces (a blending seam, a colour/sharpness mismatch between inside and outside the mask, slight misalignment), whatever generator made the face. SBI fakes those traces *without any generator*: take a real image, make a slightly altered copy of it, and blend the copy back in through a face-shaped mask. Label = fake. The model can only tell these apart from reals by the blending traces, which is exactly the method-independent signal we want.

### 3.1 Generation (`make_sbi(img, rng) -> (img_fake, mask)`)
1. `source = img`. `target = img` with, randomly:
   - colour: RGB shift ±20, hue/saturation ±10, brightness/contrast ±0.1;
   - frequency: one of {sharpen, Gaussian blur σ 0.5-1.5, downscale x0.5-0.9 then upscale};
   - geometry: translate ±3%, scale 0.95-1.05 (the misalignment of a pasted face).
2. **Mask:** an ellipse centred on the face prior (`mask_prior.npy` centroid, jitter ±8% of the image), axes from the prior's spread (±20%), then elastic deformation (`A.ElasticTransform` on the mask only), then a Gaussian edge blur with random kernel 5-25 px.
3. Blend ratio r ∈ {0.25, 0.5, 0.75, 1, 1, 1}: `out = r·mask·target + (1 − r·mask)·source`.
4. The output then goes through the **same** `TrainAug` pipeline (geo + colour + corruptions) as every other image, so corruptions remain label-independent.

Note: the PLAN warns these are in-the-wild photos, often with several people, so the face isn't always centred. If the audit grid (§3.3) shows the prior-ellipse missing faces too often, go to 3.2.

### 3.2 Gray-area option (needs your decision on 6 Nov, only if 3.1 fails the audit)
OpenCV ships a classical **Haar-cascade face detector** inside the `opencv-python-headless` package we already have (no download). It's a generic 2001-era face detector, not a deepfake detector, and it would be used on **training reals only** (never at inference, never on test). PLAN §2 rates general face detectors "allowed in principle, avoid if pair/elliptical masks suffice". Default: **not used**. I'll ask before switching it on.

### 3.3 Mixing into training
- `SBIDataset(df_real_train, ratio)`: a virtual dataset of `ratio x n_real` items; each item picks a real image (deterministic per epoch and worker seed) and synthesises a fresh SBI fake. It's concatenated with the normal `FaceDataset`. New pseudo-fakes every epoch = effectively unlimited fakes.
- Default ratio 0.3 (~1,050 SBI per epoch vs 2,800 real fakes per fold).
- Audit: `outputs/audit/sbi_examples.jpg`, 32 examples (source | SBI | mask). Show the user before training on it.

---

## 4. Pair-aware batching (`src/dfd/data.py`, ~1 h)

`GroupBatchSampler(df, batch_size, max_per_group=4)`: shuffle groups; fill each batch by adding *all* members of the next group (up to 4); singletons fill the rest. When the real and fake of one photo sit in the same batch, scene-level cues ("this photo looks like a fake-class photo") can't reduce the loss for both, so the gradient pushes toward the face difference.
Expectation: small. Real-fake pairs touch only ~5% of the images. It's cheap to test, and it's the first to cut.

---

## 5. Experiments (B0, P2 recipe, 3 folds unless stated)

| run | config diff (vs `p2_final`) | hypothesis | expected effect | time |
|---|---|---|---|---|
| `p4_maskhead` | `model.mask_head: true`, `loss.mask_weight: 0.2`, `data.pair_masks: true` | Localisation supervision focuses features on the face | clean +0.003-0.008, held-out + | 12 min |
| `p4_pairbatch` | `data.group_batches: true` | Within-batch pairs suppress scene cues | ±0.003 (tie likely) | 12 min |
| `p4_sbi03` | `sbi: {ratio: 0.3}` | Method-independent blending cue | clean ±0.005, LOCO (§6) +0.02-0.05 | 13 min |
| `p4_sbi06` | `sbi: {ratio: 0.6}` | More pseudo-fakes; risk: model drifts toward SBI-only cues | LOCO ≥ sbi03, clean may drop | 14 min |
| `p4_sbi_mask` | `sbi: {ratio: 0.3}` + mask head (SBI provides exact masks) | The two reinforce each other (Face X-ray + SBI) | **best expected**: clean +0.005-0.01, LOCO +0.03+ | 14 min |
| `p4_sbi_only` (diagnostic, fold 0) | real fakes **removed** from training; reals + SBI only | If AUC on *real* val fakes is well above 0.5 (e.g. > 0.65), SBI captures signal that transfers to real fakes | n/a | 4 min |
| `p4_maskw` | best of above with `mask_weight: 0.1` and `0.5` | weight sensitivity | tie expected | 24 min |

Decision rule: as in PHASE2 §1.2 (paired 3-fold, Δ ≥ 0.004 on all 3 folds), **plus** the LOCO criterion: SBI is adopted if LOCO AUC improves by ≥ 0.01 **and** the selection score does not drop by more than 0.004. We accept a tiny clean-AUC cost for a real gain on unseen methods, because the organisers say explicitly that test contains unseen methods.

Total B0 GPU ≈ 1.5 h + LOCO 0.5 h.

---

## 6. Checks against over-fitting to pair artefacts

1. **Mask audit** (§1, human check). Bad masks that cover corruption differences instead of the face would teach the head to look for corruptions.
2. **Pair vs non-pair breakdown.** For every Phase 4 run, report val AUC separately on (all reals + fakes that belong to a real-fake pair group) and (all reals + fakes in no pair group). Grouped folds already guarantee that a val pair's partner was never in training. If a gain shows up **only** on the pair subset, the model learned something specific to our ~170 pairs → reject. The gain must hold on the non-pair subset (~95% of fakes).
3. **Leave-one-cluster-out (LOCO): our unseen-method proxy.** Cluster the *training* fakes into k=4 groups by ImageNet `convnext_tiny` embeddings (reuse `groups.py`'s embedding code; train images only). For each cluster c: train on fold-0-train **minus cluster-c fakes**, then evaluate AUC on fold-0-val reals vs fold-0-val cluster-c fakes. Run for `p2_final` vs the best SBI config: 4 clusters x 2 configs x ~4 min ≈ 32 min. Honest caveat for the report: ImageNet-feature clusters partly reflect scene content, not only the generation method. It's a proxy, not proof.
   **Implemented (9 Nov, `src/dfd/loco.py`):** `python -m dfd.loco clusters` (KMeans k=4 on the cached L2-normalised embeddings, n_init 10, seed 42, clusters renumbered by size) -> `data/loco_clusters.csv` (committed; reals carry their nearest centroid) + `outputs/audit/loco_clusters.jpg`. Fake cluster sizes 1080/929/907/606; fold-0 val fakes 214/186/200/122. The reals spread over the same centroids almost like the fakes (total-variation distance 0.12), and the grid shows cluster 0 = heavily corrupted images. So these are mostly content/corruption clusters and the caveat above is real. Training hook: config key `data.loco_holdout: c` (train.py drops the train fakes of cluster c; val unchanged; refused with `--fold full`). `python -m dfd.loco configs --base configs/p2/p2_final.yaml` writes `configs/loco/p2_final_loco{c}.yaml` + `queues/loco_p2_final.yaml` (fold 0, `--no_log --valc_limit 128`); run the same command with the SBI config for the second recipe. `python -m dfd.loco report --names p2_final <sbi_name>` -> `outputs/loco/loco_report.{md,json}`: per cluster, AUC held-out (all val reals vs cluster-c val fakes), a content-matched variant (only the val reals nearest to centroid c), AUC on the seen clusters, the reference run (the recipe's normal fold-0 run) on the same subset, and gap = ref - held-out.
4. **SBI-only diagnostic** (§5): does the pseudo-fake signal transfer to real fakes at all?
5. **Grad-CAM spot check** (build `src/dfd/gradcam.py` here; reused in Phase 6): 16 val fakes, `p2_final` vs best Phase 4 model. Heatmaps should move *onto* the face and its boundary, not onto snow, patches or background.

---

## 7. Port to big backbones (overnight 7→8 Nov, ~2.5-3.5 h)
Re-run the 1-2 best Phase 3 backbones with the adopted Phase 4 additions on **5 folds** (e.g. ConvNeXt-S ~1.1 h, B4@320 ~1.8 h with SBI CPU overhead). Keep the Phase 3 versions too: the Phase 5 pool may contain both (with/without SBI). They are somewhat diverse, but they're the same architecture, so they don't count toward the "diverse architectures" bonus.

---

## 8. Go / no-go and cuts
- **Go to Phase 5** on 9 Nov morning regardless. Whatever is adopted by then is in; anything not adopted is written up as an ablation (good report material either way).
- If SBI hurts both clean and LOCO: drop it, report it honestly ("SBI did not transfer to this dataset's in-the-wild swaps; likely because ...").
- **Cut order if late:** `p4_maskw` → pair batching → LOCO (keep pair/non-pair breakdown) → porting to the 2nd backbone → mask head. Keep SBI + its audit, since it's the most promising robustness lever.

## 9. Risks
| risk | mitigation |
|---|---|
| Face prior ellipse misses faces in multi-person crops → SBI fakes blend background | Audit grid; Haar option (with your OK); even imperfect SBI teaches "blend seam = fake" |
| Few pair masks (≈150) → mask head barely trained on real fakes | Reals' zero-masks + SBI exact masks supply most supervision |
| SBI CPU cost slows epochs | SBI ≈ 1-3 ms/img on 12 workers: negligible for B0; measured in the smoke test |
| Learning pair-specific artefacts | §6 checks 2-3 |
| VRAM: mask head adds activations | Small (stride-16, 64 channels); B4 at batch 16 has ~4 GB of headroom |

## Phase 4 exit criteria
- [ ] Pair masks + SBI examples audited with the user; thresholds recorded in `masks_meta.csv` / config.
- [ ] B0 ablation table (maskhead, pairbatch, SBI ratios, SBI+mask, SBI-only, LOCO, pair/non-pair breakdown) logged in EXPERIMENTS.md with verdicts.
- [ ] Adopted additions ported to 1-2 big backbones with 5-fold OOF + `valc_probs.npz`.
- [ ] Grad-CAM before/after figure saved in `outputs/figures/` (report material).
