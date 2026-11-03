# Experiments log

Every training run is appended here. Metrics are on grouped-CV validation (see `data/folds.csv`).
Selection score = 0.5 x clean AUC + 0.5 x mean val-C AUC (all corruptions).

| date | run name | model | key changes | clean AUC | val-C seen | val-C held-out | worst | selection score | notes |
|---|---|---|---|---|---|---|---|---|---|
