"""Regenerate the technical-report figures into docs/report/fig/ (CPU is enough).

  CUDA_VISIBLE_DEVICES="" DF_DATA_ROOT=<hackathon_dataset> .venv/bin/python scripts/make_figures.py [--only curves valc arch gradcam]

  curves.png        train loss / val AUC of the three v1_ens members (fold 0)  (same plotting as scripts/plot_curves.py)
  valc_chart.png    val-C AUC per corruption (5-fold mean per member) + clean / val-C / OOF summary incl. the ensemble
  architecture.png  inference pipeline of inference_script.py + outputs/submissions/v1_ens/ensemble.json
  gradcam_grid.png  4 real + 4 fake fold-0 VALIDATION images x 3 members (equivalent to `python -m dfd.gradcam compare`,
                    but class-balanced); single-model grids: python -m dfd.gradcam grid --ckpt <run>/fold0/last.pt --per_class 4
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNS = ROOT / "outputs" / "runs"
OUT = ROOT / "docs" / "report" / "fig"
ENS = json.load(open(ROOT / "outputs" / "submissions" / "v1_ens" / "ensemble.json"))
MEMBERS = [m["run"] for m in ENS["members"]]
LABELS = {"p2_final": "EfficientNet-B0 @224", "b4_320_final": "EfficientNet-B4 @320", "vitb_final": "ViT-B/16 @224"}
# Okabe-Ito colour-blind-safe palette
COL = {"p2_final": "#0072B2", "b4_320_final": "#E69F00", "vitb_final": "#009E73", "ens": "#CC79A7"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in MEMBERS:
        m = pd.read_csv(RUNS / r / "fold0" / "metrics.csv")
        c = COL[r]
        ax1.plot(m.epoch, m.train_loss, "--", color=c, label=f"{LABELS[r]} train")
        ax1.plot(m.epoch, m.val_loss, "-", color=c, label=f"{LABELS[r]} val")
        ax2.plot(m.epoch, m.val_auc, "-o", ms=3, color=c, label=LABELS[r])
    ax1.set(xlabel="epoch", ylabel="BCE loss", title="Loss (dashed = train, solid = validation), fold 0")
    ax2.set(xlabel="epoch", ylabel="validation AUC-ROC", title="Validation AUC (fold 0, raw weights)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=170)
    plt.close(fig)


def valc():
    per = {}
    for r in MEMBERS:
        rows = []
        for f in range(5):
            v = json.load(open(RUNS / r / f"fold{f}" / "eval.json"))["valc"]
            rows.append({k: np.mean(list(s.values())) for k, s in v.items()})
        per[r] = pd.DataFrame(rows).mean()
    df = pd.DataFrame(per)
    held = ["motion_blur", "shot_noise", "speckle_noise", "fog", "sketch", "posterise"]
    df = df.loc[df.mean(axis=1).sort_values().index]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
    y = np.arange(len(df)); h = 0.26
    for i, r in enumerate(MEMBERS):
        ax.barh(y + (i - 1) * h, df[r], h, color=COL[r], label=LABELS[r])
    ax.set_yticks(y)
    ax.set_yticklabels([k + (" *" if k in held else "") for k in df.index])
    ax.set_xlim(0.6, 0.97); ax.set_xlabel("AUC-ROC (mean over severities 1-3 and 5 folds)")
    ax.set_title("Corrupted validation (val-C) AUC per corruption\n(* = held out from training augmentation)")
    ax.grid(axis="x", alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    # summary panel
    s = ENS["singles"]
    names = ["OOF AUC\n(clean)", "val-C AUC\n(mean)"]
    groups = [(r, [s[r]["oof_auc"], s[r]["valc_auc"]]) for r in MEMBERS] + \
             [("ens", [ENS["metrics"]["oof_auc"], ENS["metrics"]["valc_auc"]])]
    w = 0.2
    for i, (r, vals) in enumerate(groups):
        xs = np.arange(2) + (i - 1.5) * w
        b = ax2.bar(xs, vals, w, color=COL[r], label="Ensemble (equal weights)" if r == "ens" else LABELS[r])
        for x, v in zip(xs, vals):
            ax2.text(x, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(names)
    ax2.set_ylim(0.84, 0.95); ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Pooled 5-fold OOF AUC and val-C AUC")
    ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "valc_chart.png", dpi=170)
    plt.close(fig)


def arch():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#444", fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))

    box(0.1, 2.2, 1.8, 1.2, "Face image\n224x224 JPEG\n(test_dir/*.jpg)", "#FFFFFF")
    box(2.4, 2.2, 2.3, 1.2, "Decode (OpenCV, RGB)\nINTER_AREA to 224\nuint8 CHW, batch 64\nno flip TTA", "#E8EEF7", fs=8.5)
    arrow(1.9, 2.8, 2.4, 2.8)
    ys = [4.0, 2.2, 0.4]
    spec = [("p2_final", "EfficientNet-B0 (ImageNet)\ninput 224 | 1 logit"),
            ("b4_320_final", "EfficientNet-B4 ra2_in1k\nbilinear up to 320 | 1 logit"),
            ("vitb_final", "ViT-B/16 augreg2 in21k->in1k\ninput 224 | 1 logit")]
    for y, (r, t) in zip(ys, spec):
        m = next(m for m in ENS["members"] if m["run"] == r)
        box(5.2, y, 3.0, 1.2, t + "\nFP16 autocast", COL[r] + "33", COL[r], fs=8.5)
        arrow(4.7, 2.8, 5.2, y + 0.6)
        box(8.6, y + 0.2, 1.7, 0.8, f"logit / T\nT = {m['temperature']:.3f}\nw = {m['weight']:.3f}", "#FFFFFF", COL[r], fs=8)
        arrow(8.2, y + 0.6, 8.6, y + 0.6)
        arrow(10.3, y + 0.6, 10.75, 2.8)
    box(10.75, 2.2, 1.35, 1.2, "weighted sum\nsigmoid(s / T_ens)\nT_ens = %.3f" % ENS["temperature_ens"], "#FBE9F2", COL["ens"], fs=8)
    arrow(12.1, 2.8, 12.35, 2.8)
    box(12.35, 2.2, 0.6, 1.2, "P(fake)", "#FFFFFF", "#222", fs=8, bold=True)
    ax.text(11.7, 1.7, f"label = 1 if P >= {ENS['threshold']:.3f}\n(OOF F1-optimal threshold)", ha="center", va="top", fontsize=8)
    ax.text(6.5, 5.45, "v1_ens inference pipeline (inference_script.py, single weights bundle, equal-weight logit ensemble)",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.1, 0.4, "Normalisation (ImageNet mean/std) runs inside each member;\nconfidence column = P(fake) in [0, 1].", fontsize=8, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=170)
    plt.close(fig)


def gradcam():
    from PIL import Image
    from dfd.paths import resolve_data_root
    from dfd.gradcam import cams_for, fold_oof, load_ckpt, overlay, read_rgb224
    from dfd.viz import _tile, grid_from_cells
    root = resolve_data_root(None)
    ckpts = [RUNS / r / "fold0" / "last.pt" for r in MEMBERS]
    models = [load_ckpt(c, "cpu") for c in ckpts]
    oof = fold_oof(ckpts[0], models[0][1])
    per = 4
    sel = pd.concat([oof[oof.label == 0].sample(per, random_state=3), oof[oof.label == 1].sample(per, random_state=3)])
    imgs = [read_rgb224(root, r.image_id, int(r.label)) for r in sel.itertuples()]
    cells = [_tile(Image.fromarray(im), "val " + ("fake" if l else "real"), (200, 40, 40) if l else (40, 120, 60), size=160)
             for im, l in zip(imgs, sel.label)]
    for (m, _), r in zip(models, MEMBERS):
        cams, probs = cams_for(m, imgs, "pred", "cpu")
        cells += [_tile(Image.fromarray(overlay(im, cm)), f"{LABELS[r].split(' @')[0]} p={p:.2f}", (60, 60, 60), size=160)
                  for im, cm, p in zip(imgs, cams, probs)]
    grid_from_cells(cells, OUT / "gradcam_grid.png", "Grad-CAM (evidence for the predicted class), fold-0 validation images: "
                    "4 real + 4 fake; rows: image, B0, B4@320, ViT-B/16", cols=len(imgs), gap=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["curves", "valc", "arch", "gradcam"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in a.only:
        {"curves": curves, "valc": valc, "arch": arch, "gradcam": gradcam}[k]()
        print("done", k)
