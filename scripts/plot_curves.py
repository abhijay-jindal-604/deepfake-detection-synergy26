"""Plot loss and AUC curves from one or more metrics.csv files.

  python scripts/plot_curves.py outputs/runs/baseline_b0/fold0 [more run dirs] --out curves.png
  (a <run>/full dir has only train_loss: plotted on the loss panel, AUC panel skipped for it)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for r in a.runs:
    m = pd.read_csv(Path(r) / "metrics.csv")
    name = "/".join(Path(r).parts[-2:])
    l, = ax1.plot(m.epoch, m.train_loss, "--", label=f"{name} train")
    if "val_loss" in m:   # --fold full runs have no validation columns
        ax1.plot(m.epoch, m.val_loss, "-", color=l.get_color(), label=f"{name} val")
    if "val_auc" in m:
        ax2.plot(m.epoch, m.val_auc, "-o", ms=3, color=l.get_color(), label=name)
ax1.set(xlabel="epoch", ylabel="BCE loss", title="loss (dashed = train, solid = val)")
ax2.set(xlabel="epoch", ylabel="val AUC", title="validation AUC")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(a.out, dpi=120)
print("wrote", a.out)
