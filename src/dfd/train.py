"""Train one fold (or all folds) from a YAML config.

  python -m dfd.train --config configs/baseline_b0.yaml --fold 0 --data_root <root>
  python -m dfd.train --config configs/baseline_b0.yaml --fold all
  smoke test: --limit 512 --epochs 1 --valc_limit 64 --no_log
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from .augment import TrainAug
from .data import FaceDataset, load_folds, make_loader
from .evaluate import clean_metrics, full_eval, predict
from .models import build_model, param_groups
from .paths import REPO_ROOT, RUNS_DIR, resolve_data_root
from .seed import SEED, seed_everything

EXPERIMENTS_MD = REPO_ROOT / "docs" / "EXPERIMENTS.md"


def lr_lambdas(groups, steps_per_epoch: int, epochs: int, warmup_epochs: float, min_lr: float):
    total = steps_per_epoch * epochs
    warm = max(1, int(steps_per_epoch * warmup_epochs))

    def make(base_lr):
        floor = min_lr / base_lr

        def f(step):
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

        return f

    return [make(g["lr"]) for g in groups]


def train_fold(cfg: dict, fold: int, root: Path, run_dir: Path, args) -> dict:
    seed_everything(SEED, deterministic=cfg.get("deterministic", False))
    device = "cuda"
    out = run_dir / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    folds = load_folds()
    df_tr = folds[folds.fold != fold].reset_index(drop=True)
    df_va = folds[folds.fold == fold].reset_index(drop=True)
    if args.limit:
        df_tr = df_tr.sample(n=min(args.limit, len(df_tr)), random_state=SEED).reset_index(drop=True)
        df_va = df_va.sample(n=min(args.limit // 2, len(df_va)), random_state=SEED).reset_index(drop=True)

    t = cfg["train"]
    workers = t.get("workers", 12)
    train_loader = make_loader(FaceDataset(df_tr, root, TrainAug(cfg)), t["batch_size"], True, workers)
    val_loader = make_loader(FaceDataset(df_va, root), t["batch_size"] * 2, False, workers)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    o = cfg["optim"]
    groups = param_groups(model, o["lr_backbone"], o["lr_head"], o["weight_decay"])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    epochs = t["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambdas(groups, len(train_loader), epochs, o.get("warmup_epochs", 1), o.get("min_lr", 1e-6))
    )
    ls = cfg.get("loss", {}).get("label_smoothing", 0.0)
    bce = torch.nn.BCEWithLogitsLoss()

    tb = SummaryWriter(out / "tb")
    rows, best_auc, best_epoch = [], -1.0, -1
    print(f"[fold {fold}] train {len(df_tr)} | val {len(df_va)} | {len(train_loader)} steps/epoch | {epochs} epochs")
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            y_s = y * (1 - ls) + 0.5 * ls
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = bce(logits.float(), y_s)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if o.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            opt.step()
            sched.step()
            losses.append(loss.item())
        train_time = time.time() - t0

        p, yv = predict(model, val_loader, device)
        val_loss = float(torch.nn.functional.binary_cross_entropy(torch.tensor(p).clamp(1e-7, 1 - 1e-7), torch.tensor(yv)))
        m = clean_metrics(yv, p)
        row = {
            "epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_auc": m["auc"],
            "val_f1": m["f1@0.5"], "val_acc": m["acc@0.5"], "lr_backbone": opt.param_groups[0]["lr"],
            "epoch_time_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        for k, v in row.items():
            if k != "epoch":
                tb.add_scalar(k, v, ep)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        flag = ""
        ckpt = {"state_dict": model.state_dict(), "config": cfg, "fold": fold, "epoch": ep, "val_auc": m["auc"]}
        if m["auc"] > best_auc:
            best_auc, best_epoch, flag = m["auc"], ep, " *best"
            torch.save(ckpt, out / "best.pt")
        torch.save(ckpt, out / "last.pt")
        print(
            f"ep {ep:2d} | train {row['train_loss']:.4f} | val {val_loss:.4f} | AUC {m['auc']:.4f} | "
            f"F1 {m['f1@0.5']:.4f} | lr {row['lr_backbone']:.2e} | {train_time:.0f}s train{flag}",
            flush=True,
        )
    tb.close()

    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    p, yv = predict(model, val_loader, device)
    pd.DataFrame({"image_id": df_va.image_id, "label": yv.astype(int), "prob": p}).to_csv(out / f"oof_fold{fold}.csv", index=False)
    print(f"\n[fold {fold}] best epoch {best_epoch} AUC {best_auc:.4f} -> val-C on best checkpoint")
    res = full_eval(model, df_va, root, out, t["batch_size"] * 2, workers, args.valc_limit)
    res["best_epoch"] = best_epoch
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    return res


def log_experiment(run: str, cfg: dict, s: dict, notes: str) -> None:
    row = (
        f"| {dt.date.today().isoformat()} | {run} | {cfg['model']['name']} | {cfg.get('description', '')} | "
        f"{s['clean_auc']:.4f} | {s['valc_seen']:.4f} | {s['valc_heldout']:.4f} | {s['worst_auc']:.4f} "
        f"({s['worst_setting']}) | {s['selection_score']:.4f} | {notes} |\n"
    )
    with open(EXPERIMENTS_MD, "a") as f:
        f.write(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="0", help="fold index or 'all'")
    ap.add_argument("--data_root")
    ap.add_argument("--run_name", help="default: config file stem")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="subsample train (smoke test)")
    ap.add_argument("--valc_limit", type=int, help="subsample val images for val-C (smoke test)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no_log", action="store_true", help="do not append to EXPERIMENTS.md")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg = copy.deepcopy(cfg)
        cfg["train"]["epochs"] = args.epochs
    run = args.run_name or Path(args.config).stem
    run_dir = RUNS_DIR / run
    folds = range(5) if args.fold == "all" else [int(args.fold)]

    results = {}
    for k in folds:
        results[k] = train_fold(cfg, k, root, run_dir, args)
        if not args.no_log:
            log_experiment(f"{run}/fold{k}", cfg, results[k]["summary"], f"best ep {results[k]['best_epoch']}. {args.notes}".strip())

    if len(results) == 5:
        oof = pd.concat(pd.read_csv(run_dir / f"fold{k}" / f"oof_fold{k}.csv") for k in folds)
        oof.to_csv(run_dir / "oof_all.csv", index=False)
        aucs = [results[k]["summary"]["clean_auc"] for k in folds]
        msg = f"OOF AUC (all 7000): {roc_auc_score(oof.label, oof.prob):.4f} | per-fold mean {np.mean(aucs):.4f} +- {np.std(aucs):.4f}"
        print("\n" + msg)
        (run_dir / "oof_summary.txt").write_text(msg + "\n")
        if not args.no_log:
            with open(EXPERIMENTS_MD, "a") as f:
                f.write(f"| {dt.date.today().isoformat()} | {run}/OOF | {cfg['model']['name']} | 5-fold summary | {msg} | | | | | |\n")


if __name__ == "__main__":
    main()
