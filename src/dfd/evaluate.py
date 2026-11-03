"""Clean + val-C evaluation of a checkpoint on a fold's validation split.

  python -m dfd.evaluate --ckpt outputs/runs/<run>/fold0/best.pt --data_root <root>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from .corruptions import ALL, HELD_OUT, SEEN, SEVERITIES
from .data import FaceDataset, ValCDataset, load_folds, make_loader
from .models import build_model
from .paths import resolve_data_root


@torch.inference_mode()
def predict(model, loader, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def clean_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = int(np.argmax(f1s[:-1])) if len(thr) else 0
    return {
        "auc": float(roc_auc_score(y, p)),
        "f1@0.5": float(f1_score(y, p >= 0.5)),
        "acc@0.5": float(accuracy_score(y, p >= 0.5)),
        "best_threshold": float(thr[best]) if len(thr) else 0.5,
        "f1@best": float(f1s[best]),
    }


def evaluate_valc(model, df_val: pd.DataFrame, root: Path, batch_size: int, workers: int, device: str = "cuda") -> dict:
    settings = [(n, s) for n in ALL for s in SEVERITIES]
    ds = ValCDataset(df_val, root, settings)
    probs, labels = predict(model, make_loader(ds, batch_size, train=False, workers=workers), device)
    n = len(df_val)
    res = {}
    for k, (name, sev) in enumerate(settings):
        sl = slice(k * n, (k + 1) * n)
        res.setdefault(name, {})[str(sev)] = float(roc_auc_score(labels[sl], probs[sl]))
    return res


def summarise(clean: dict, valc: dict) -> dict:
    def mean_of(names):
        return float(np.mean([v for n in names for v in valc[n].values()]))

    all_vals = [v for n in valc for v in valc[n].values()]
    worst_name, worst_sev = min(((n, s) for n in valc for s in valc[n]), key=lambda t: valc[t[0]][t[1]])
    mean_all = float(np.mean(all_vals))
    return {
        "clean_auc": clean["auc"],
        "valc_seen": mean_of(SEEN),
        "valc_heldout": mean_of(HELD_OUT),
        "valc_all": mean_all,
        "worst_auc": float(min(all_vals)),
        "worst_setting": f"{worst_name}@{worst_sev}",
        "selection_score": 0.5 * clean["auc"] + 0.5 * mean_all,
    }


def format_report(clean: dict, valc: dict, summary: dict) -> str:
    lines = [
        f"clean: AUC {clean['auc']:.4f} | F1@0.5 {clean['f1@0.5']:.4f} | acc@0.5 {clean['acc@0.5']:.4f} | "
        f"best thr {clean['best_threshold']:.3f} -> F1 {clean['f1@best']:.4f}",
        "",
        f"{'corruption':<22}{'s1':>8}{'s2':>8}{'s3':>8}{'mean':>8}",
    ]
    for name in ALL:
        v = [valc[name][str(s)] for s in SEVERITIES]
        tag = " *" if name in HELD_OUT else ""
        lines.append(f"{name + tag:<22}" + "".join(f"{x:>8.4f}" for x in v) + f"{np.mean(v):>8.4f}")
    lines += [
        "(* = held out from training augmentation)",
        "",
        f"mean val-C seen {summary['valc_seen']:.4f} | held-out {summary['valc_heldout']:.4f} | all {summary['valc_all']:.4f}",
        f"worst {summary['worst_auc']:.4f} ({summary['worst_setting']})",
        f"SELECTION SCORE = 0.5*clean + 0.5*valC_all = {summary['selection_score']:.4f}",
    ]
    return "\n".join(lines)


def full_eval(model, df_val, root, out_dir: Path, batch_size: int = 128, workers: int = 12, valc_limit: int | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_loader(FaceDataset(df_val, root), batch_size, train=False, workers=workers)
    p, y = predict(model, loader, device)
    clean = clean_metrics(y, p)
    dv = df_val if valc_limit is None else df_val.sample(n=min(valc_limit, len(df_val)), random_state=42)
    valc = evaluate_valc(model, dv, root, batch_size, workers, device)
    summary = summarise(clean, valc)
    report = format_report(clean, valc, summary)
    print(report)
    out = {"clean": clean, "valc": valc, "summary": summary}
    (out_dir / "eval.json").write_text(json.dumps(out, indent=2))
    (out_dir / "eval.txt").write_text(report + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["config"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).to(memory_format=torch.channels_last)
    folds = load_folds()
    df_val = folds[folds.fold == ck["fold"]].reset_index(drop=True)
    full_eval(model, df_val, root, Path(args.ckpt).parent, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
