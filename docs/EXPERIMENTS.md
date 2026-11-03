# Experiments log

Every training run is appended here. Metrics are on grouped-CV validation (see `data/folds.csv`).
Selection score = 0.5 x clean AUC + 0.5 x mean val-C AUC (all corruptions).

| date | run name | model | key changes | clean AUC | val-C seen | val-C held-out | worst | selection score | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | baseline_b0/fold0 | efficientnet_b0 | B0 baseline, basic aug only | 0.8449 | 0.7885 | 0.7208 | 0.5144 (pixelate@3) | 0.8060 | best ep 10. first baseline. BUG: head init U(-1,1) (epoch-1 train loss 2.5); superseded by zero-init run |
| 2026-11-03 | baseline_b0/fold0 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8739 | 0.8249 | 0.7710 | 0.4726 (pixelate@3) | 0.8409 | best ep 2. zero-init head |
| 2026-11-03 | b0_seencorr/fold0 | efficientnet_b0 | B0 + seen-corruption aug p=0.5 | 0.8834 | 0.8543 | 0.8384 | 0.6487 (pixelate@3) | 0.8663 | best ep 4. ablation: + seen corruptions |
| 2026-11-03 | baseline_b0_5fold/fold0 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8728 | 0.8233 | 0.7676 | 0.4760 (pixelate@3) | 0.8393 | best ep 2. 5-fold |
| 2026-11-03 | baseline_b0_5fold/fold1 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8910 | 0.8468 | 0.7907 | 0.5473 (pixelate@3) | 0.8600 | best ep 2. 5-fold |
| 2026-11-03 | baseline_b0_5fold/fold2 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8937 | 0.8553 | 0.8147 | 0.6030 (pixelate@3) | 0.8681 | best ep 12. 5-fold |
| 2026-11-03 | baseline_b0_5fold/fold3 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8697 | 0.8304 | 0.7972 | 0.5556 (pixelate@3) | 0.8448 | best ep 8. 5-fold |
| 2026-11-03 | baseline_b0_5fold/fold4 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8850 | 0.8392 | 0.7850 | 0.5356 (pixelate@3) | 0.8535 | best ep 2. 5-fold |
| 2026-11-03 | baseline_b0_5fold/OOF | efficientnet_b0 | 5-fold summary | OOF AUC (all 7000): 0.8715 | per-fold mean 0.8825 +- 0.0096 | | | | | |

## Decisions

| date | decision | reason |
|---|---|---|
| 2026-11-03 | Phase 2 code accepted | `repro_baseline_b0` fold 0 (select best, no EMA) = clean 0.8733 / selection 0.8394 vs Phase 1 0.8739 / 0.8409 (within ±0.002), best epoch 2 again |
| 2026-11-03 | EMA: keep `warmup: true` (decay 0.998 cap) | timm warmup makes the effective end decay ≈0.990 (~1 epoch of averaging) on ~1k-step runs; `warmup: false, 0.998` would leave 12-25% of the near-initial weights in the EMA |
| 2026-11-03 | Continuous JPEG severity clamped at q>=8; 4 extra look-alikes on the held-out deny-list | q5 destroyed colour entirely; AdvancedBlur/GlassBlur/Plasma* resemble motion blur / fog (stricter, none used) |
| 2026-11-03 | Step-2 weighted / continuous / extra runs based on p2_ref (p 0.5), not the winning p | lets all of steps 1-2 run in one overnight queue; winners get stacked in one combined run afterwards |
