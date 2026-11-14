#!/usr/bin/env python
"""Group bootstrap 95% CI for the ensemble's pooled out-of-fold AUC (reported in docs/report/report.md §0, §5.4).

  python scripts/bootstrap_ci.py [--ensemble outputs/submissions/v1_ens/ensemble.json] [--n 1000] [--seed 42]

Rebuilds the ensemble OOF score exactly as dfd.ensemble does (temperature-scaled logit average with the stored
weights), then resamples whole CV similarity groups (data/folds.csv `group_id`) with replacement, so real/fake
images from the same source photo always stay together. CPU only, ~1 min.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dfd.ensemble import Pool
from dfd.paths import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", type=Path, default=ROOT / "outputs/submissions/v1_ens/ensemble.json")
    ap.add_argument("--runs_dir", type=Path, default=None, help="default: runs_dir stored in the ensemble json")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ens = json.loads(args.ensemble.read_text())
    runs = [m["run"] for m in ens["members"]]
    w = np.array([m["weight"] for m in ens["members"]])
    t = np.array([m["temperature"] for m in ens["members"]])
    pool = Pool(args.runs_dir or Path(ens["runs_dir"]), runs, ens["folds"], ens["variant"])
    s, y = (w / t) @ pool.oof_z, pool.oof_y
    auc = roc_auc_score(y, s)
    print(f"point OOF AUC {auc:.4f} (ensemble.json: {ens['metrics']['oof_auc']:.4f})")

    folds = pd.read_csv(DATA_DIR / "folds.csv", dtype={"image_id": str}).set_index("image_id")
    g = folds.loc[pool.oof_ids, "group_id"].to_numpy()
    _, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
    print(f"{len(y)} images in {len(members)} groups ({sum(len(m) > 1 for m in members)} with > 1 image)")

    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.n)
    for b in range(args.n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        boot[b] = roc_auc_score(y[idx], s[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"group bootstrap ({args.n} resamples, seed {args.seed}): 95% CI [{lo:.4f}, {hi:.4f}], sd {boot.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
