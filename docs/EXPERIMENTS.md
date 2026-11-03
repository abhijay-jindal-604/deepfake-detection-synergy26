# Experiments log

Every training run is appended here. Metrics are on grouped-CV validation (see `data/folds.csv`).
Selection score = 0.5 x clean AUC + 0.5 x mean val-C AUC (all corruptions).

| date | run name | model | key changes | clean AUC | val-C seen | val-C held-out | worst | selection score | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | baseline_b0/fold0 | efficientnet_b0 | B0 baseline, basic aug only | 0.8449 | 0.7885 | 0.7208 | 0.5144 (pixelate@3) | 0.8060 | best ep 10. first baseline. BUG: head init U(-1,1) (epoch-1 train loss 2.5); superseded by zero-init run |
| 2026-11-03 | baseline_b0/fold0 | efficientnet_b0 | B0 baseline, basic aug only, zero-init head | 0.8739 | 0.8249 | 0.7710 | 0.4726 (pixelate@3) | 0.8409 | best ep 2. zero-init head |
