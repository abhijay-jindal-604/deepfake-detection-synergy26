# Phase 2 configs

| config | diff | hypothesis |
|---|---|---|
| p2_ref | base b0_seencorr, `select: last` | reference under new protocol |
| p2_ref_seed43 | `seed: 43` | DIAGNOSTIC ONLY: seed noise floor, never ships |
| p2_corr_p07 | `aug.seen_corruptions_p: 0.7` | more corruption widens coverage |
| p2_corr_p09 | `aug.seen_corruptions_p: 0.9` | too much may hurt clean AUC |
| p2_corr_weighted | `aug.corr_weights` pixelate 3, downscale 2, gaussian_noise 3, snow 2 | weak cells (pixelate@3, noise) improve |
| p2_corr_cont | `aug.sev_mode: continuous`, `sev_range: [0.5, 3.5]` | worst cell improves; held-out flat or better |
| p2_extra | `aug.extra_p: 0.3` | held-out +0.005-0.015 |
| repro_baseline_b0 | base baseline_b0, `select: best` | acceptance: reproduce Phase 1 fold 0 within ±0.002 |

p2_corr_weighted, p2_corr_cont and p2_extra are based on p2_ref for now. Re-base them on the winning `seen_corruptions_p` once the step 2 p-runs are decided (PHASE2 §3 "carry forward").
