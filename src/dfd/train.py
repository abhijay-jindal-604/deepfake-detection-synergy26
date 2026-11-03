"""Train one fold (or all folds) from a YAML config.

  python -m dfd.train --config configs/baseline_b0.yaml --fold 0 --data_root <root>
  python -m dfd.train --config configs/baseline_b0.yaml --fold all
  python -m dfd.train --config configs/p2/p2_ref.yaml --fold 0,1,2 --skip_done
  final model (PHASE5 §3): --fold full -> 100% of the data, fixed epoch count, no validation / OOF / val-C;
    writes <run>/full/last.pt (EMA weights in state_dict_ema if EMA is on), metrics.csv (train curve), done.json
  $DFD_RUNS_DIR overrides the runs directory (default outputs/runs).
  smoke test: --limit 512 --epochs 1 --valc_limit 64 --no_log
  CPU (tests while the GPU is busy): DFD_DEVICE=cpu python -m dfd.train ...  (BF16 autocast is CUDA-only)

Phase 4 keys (docs/PHASE4.md §5): model.mask_head, loss.mask_weight, data.pair_masks, data.group_batches,
data.drop_real_fakes, sbi.ratio. Without them training is bit-identical to the pre-Phase-4 code.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from timm.utils import ModelEmaV3
from torch.utils.tensorboard import SummaryWriter

from .augment import TrainAug
from .data import FaceDataset, GroupBatchSampler, build_train_set, load_folds, make_loader, train_has_masks
from .evaluate import clean_metrics, full_eval, pair_breakdown, predict
from .models import build_model, param_groups
from .paths import REPO_ROOT, RUNS_DIR, resolve_data_root
from .seed import SEED, seed_everything

EXPERIMENTS_MD = REPO_ROOT / "docs" / "EXPERIMENTS.md"


def lr_lambdas(groups, steps_per_epoch: int, epochs: int, warmup_epochs: float, min_lr: float):
    total = steps_per_epoch * epochs
    warm = max(1, int(steps_per_epoch * warmup_epochs))

    def make(g):
        base_lr = g["lr"]
        if "lr_scale" in g:  # LLRD group: decay to min_lr scaled like its LR (same end fraction as the top layer)
            floor = (min_lr * g["lr_scale"]) / base_lr
        else:
            floor = min_lr / base_lr

        def f(step):
            if step < warm:
                return (step + 1) / warm
            t = (step - warm) / max(1, total - warm)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

        return f

    return [make(g) for g in groups]


def _deep_merge(parent: dict, child: dict) -> dict:
    out = copy.deepcopy(parent)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)  # scalars and lists replace
    return out


def load_config(path, _stack: tuple = ()) -> dict:
    """YAML config with optional `base: <path relative to this file>`; child keys override the base recursively."""
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError("config base cycle: " + " -> ".join(str(p) for p in (*_stack, path)))
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    parent = load_config(path.parent / base, (*_stack, path))
    return _deep_merge(parent, cfg)


FULL = "full"  # --fold full: train on 100% of the data (PHASE5 §3), no validation, output dir <run>/full/


def parse_folds(spec: str) -> list:
    if spec == "all":
        return list(range(5))
    if spec.strip() == FULL:
        return [FULL]
    if FULL in [x.strip() for x in spec.split(",")]:
        raise ValueError("--fold full cannot be combined with other folds")
    folds = [int(x) for x in spec.split(",") if x.strip() != ""]
    if not folds or any(not 0 <= k < 5 for k in folds) or len(set(folds)) != len(folds):
        raise ValueError(f"bad --fold {spec!r}")
    return folds


def _val_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(torch.nn.functional.binary_cross_entropy(torch.tensor(p).clamp(1e-7, 1 - 1e-7), torch.tensor(y)))


def fold_dir(run_dir: Path, fold) -> Path:
    return run_dir / (FULL if fold == FULL else f"fold{fold}")


def fold_done(run_dir: Path, fold) -> bool:
    """--skip_done marker: fold k = eval.json + oof_fold{k}.csv; full = full/last.pt + full/done.json."""
    d = fold_dir(run_dir, fold)
    if fold == FULL:
        return (d / "done.json").exists() and (d / "last.pt").exists()
    return (d / "eval.json").exists() and (d / f"oof_fold{fold}.csv").exists()


def train_fold(cfg: dict, fold, root: Path, run_dir: Path, args) -> dict:
    seed = int(cfg.get("seed", SEED))
    seed_everything(seed, deterministic=cfg.get("deterministic", False))
    device = os.environ.get("DFD_DEVICE", "cuda")
    amp = device.startswith("cuda")  # BF16 autocast on CUDA only (CPU path is for tests)
    full = fold == FULL
    out = fold_dir(run_dir, fold)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    folds = load_folds()
    if full:  # 100% of the training data, no validation split (PHASE5 §3)
        df_tr = folds.reset_index(drop=True)
        df_va = folds.iloc[:0].reset_index(drop=True)
    else:
        df_tr = folds[folds.fold != fold].reset_index(drop=True)
        df_va = folds[folds.fold == fold].reset_index(drop=True)
    if args.limit:
        df_tr = df_tr.sample(n=min(args.limit, len(df_tr)), random_state=SEED).reset_index(drop=True)
        df_va = df_va.sample(n=min(args.limit // 2, len(df_va)), random_state=SEED).reset_index(drop=True)

    t = cfg["train"]
    workers = t.get("workers", 12)
    eval_bs = t.get("eval_batch_size", 128)
    accum = int(t.get("accum_steps", 1))
    select = cfg.get("select", "last")
    if select not in ("last", "best"):
        raise ValueError(f"select must be last|best, got {select!r}")
    if full and select != "last":
        print(f"[full] select={select!r} needs a validation set; using the last epoch (fixed epoch count)")
        select = "last"
    mask_head = bool(cfg["model"].get("mask_head", False))
    with_mask = train_has_masks(cfg)
    if mask_head and cfg.get("mixup"):
        raise ValueError("mixup + model.mask_head is not supported (mixed images have no mask target)")
    train_ds, df_fit, sbi_ds = build_train_set(cfg, df_tr, root, TrainAug(cfg), seed)
    sampler = None
    if (cfg.get("data") or {}).get("group_batches"):
        sampler = GroupBatchSampler(df_fit, t["batch_size"], (cfg.get("data") or {}).get("max_per_group", 4), seed,
                                    n_extra=len(sbi_ds) if sbi_ds is not None else 0)
    train_loader = make_loader(train_ds, t["batch_size"], True, workers, seed=seed, batch_sampler=sampler)
    val_loader = None if full else make_loader(FaceDataset(df_va, root), eval_bs, False, workers, seed=seed)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    if t.get("grad_ckpt"):
        model.net.set_grad_checkpointing(True)
    o = cfg["optim"]
    groups = param_groups(model, o["lr_backbone"], o["lr_head"], o["weight_decay"], o.get("layer_decay"))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    epochs = t["epochs"]
    n_micro = len(train_loader)
    steps_per_epoch = math.ceil(n_micro / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambdas(groups, steps_per_epoch, epochs, o.get("warmup_epochs", 1), o.get("min_lr", 1e-6))
    )
    ls = cfg.get("loss", {}).get("label_smoothing", 0.0)
    mask_w = float(cfg.get("loss", {}).get("mask_weight", 0.2)) if mask_head else 0.0
    bce = torch.nn.BCEWithLogitsLoss()

    # EMA of weights + buffers (BN running stats are lerped too; num_batches_tracked is copied)
    e = cfg.get("ema") or None
    ema = None
    if e:
        ema = ModelEmaV3(model, decay=e.get("decay", 0.998), use_warmup=e.get("warmup", True))
    # mixup has its own RNG so that mixup-off consumes no randomness anywhere
    mx = cfg.get("mixup") or None
    mix_rng = np.random.default_rng(seed + 7919) if mx else None

    tb = SummaryWriter(out / "tb")
    rows = []
    best_score, best_epoch = -1.0, -1
    opt_steps = 0
    if with_mask:
        print(
            f"[fold {fold}] phase4: mask_head {mask_head} (w {mask_w}) | pair masks "
            f"{sum(m is not None for m in getattr(train_ds, 'datasets', [train_ds])[0].mask_paths)} | "
            f"sbi {len(sbi_ds) if sbi_ds is not None else 0}/epoch | fit images {len(df_fit)} | group_batches {sampler is not None}"
        )
    elif sampler is not None:
        print(f"[fold {fold}] phase4: group_batches on")
    print(
        f"[fold {fold}] train {len(df_tr)} | val {len(df_va)} | {n_micro} batches/epoch, accum {accum} -> "
        f"{steps_per_epoch} steps | {epochs} epochs | select {select} | ema {bool(ema)} | mixup {bool(mx)} | seed {seed}"
    )
    for ep in range(1, epochs + 1):
        model.train()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)  # SBI: fresh pseudo-fakes per epoch (shared-memory, reaches persistent workers)
        if sampler is not None:
            sampler.set_epoch(ep)
        t0, losses, mask_losses = time.time(), [], []
        for i, batch in enumerate(train_loader):
            x, y = batch[0], batch[1]
            g0 = (i // accum) * accum
            group = min(accum, n_micro - g0)  # last partial group of the epoch is smaller
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            y_s = y * (1 - ls) + 0.5 * ls
            if mx and mix_rng.random() < mx.get("p", 1.0):
                lam = float(mix_rng.beta(mx["alpha"], mx["alpha"]))
                perm = torch.from_numpy(mix_rng.permutation(x.shape[0])).to(device)
                x = (lam * x.float() + (1 - lam) * x.float()[perm]).contiguous(memory_format=torch.channels_last)
                y_s = lam * y_s + (1 - lam) * y_s[perm]
            with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.bfloat16, enabled=amp):
                out_ = model(x)
            if mask_head:
                logits, mask_logits = out_
                loss = bce(logits.float(), y_s)
                mk = batch[2].to(device, non_blocking=True)
                mv = batch[3].to(device, non_blocking=True)
                # area-downsample the target to the feature grid (14x14 at 224, 20x20 for B4@320)
                mk = torch.nn.functional.adaptive_avg_pool2d(mk[:, None], mask_logits.shape[-2:])[:, 0]
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    mask_logits.float(), mk, reduction="none").mean(dim=(1, 2))
                mask_loss = (per * mv).sum() / mv.sum().clamp(min=1.0)  # valid samples only; 0 if none
                loss = loss + mask_w * mask_loss
                mask_losses.append(mask_loss.item())
            else:
                logits = out_
                loss = bce(logits.float(), y_s)
            if i == g0:
                opt.zero_grad(set_to_none=True)
            (loss / group if group > 1 else loss).backward()
            if i == g0 + group - 1:
                if o.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
                opt.step()
                sched.step()
                opt_steps += 1
                if ema is not None:
                    ema.update(model, step=opt_steps)
            losses.append(loss.item())
        train_time = time.time() - t0

        if full:  # no validation: log the training curve (PHASE5 §3 curve-match check), save last.pt
            row = {"epoch": ep, "train_loss": float(np.mean(losses))}
            if mask_head:
                row["train_mask_loss"] = float(np.mean(mask_losses))
            row.update({"lr_backbone": opt.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1)})
            rows.append(row)
            for k, v in row.items():
                if k != "epoch":
                    tb.add_scalar(k, v, ep)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            ckpt = {"state_dict": model.state_dict(), "config": cfg, "fold": FULL, "epoch": ep}
            if ema is not None:
                ckpt["state_dict_ema"] = ema.module.state_dict()
            tmp = out / "last.pt.tmp"
            torch.save(ckpt, tmp)
            tmp.replace(out / "last.pt")
            print(f"ep {ep:2d} | train {row['train_loss']:.4f} | lr {row['lr_backbone']:.2e} | {train_time:.0f}s train",
                  flush=True)
            continue

        p, yv = predict(model, val_loader, device)
        m = clean_metrics(yv, p)
        row = {
            "epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": _val_loss(p, yv), "val_auc": m["auc"],
            "val_f1": m["f1@0.5"], "val_acc": m["acc@0.5"],
        }
        if mask_head:  # train_loss = cls + mask_weight * mask
            row["train_mask_loss"] = float(np.mean(mask_losses))
        if ema is not None:
            pe, _ = predict(ema.module, val_loader, device)
            me = clean_metrics(yv, pe)
            row.update({"val_loss_ema": _val_loss(pe, yv), "val_auc_ema": me["auc"], "val_f1_ema": me["f1@0.5"]})
        row.update({"lr_backbone": opt.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1)})
        rows.append(row)
        for k, v in row.items():
            if k != "epoch":
                tb.add_scalar(k, v, ep)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        sel_auc = row["val_auc_ema"] if ema is not None else row["val_auc"]
        ckpt = {"state_dict": model.state_dict(), "config": cfg, "fold": fold, "epoch": ep, "val_auc": m["auc"]}
        if ema is not None:
            ckpt["state_dict_ema"] = ema.module.state_dict()
            ckpt["val_auc_ema"] = row["val_auc_ema"]
        flag = ""
        if sel_auc > best_score:
            best_score, best_epoch, flag = sel_auc, ep, " *best"
            torch.save(ckpt, out / "best.pt")
        torch.save(ckpt, out / "last.pt")
        ema_txt = f" | EMA AUC {row['val_auc_ema']:.4f} val {row['val_loss_ema']:.4f}" if ema is not None else ""
        print(
            f"ep {ep:2d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | AUC {m['auc']:.4f} | "
            f"F1 {m['f1@0.5']:.4f}{ema_txt} | lr {row['lr_backbone']:.2e} | {train_time:.0f}s train{flag}",
            flush=True,
        )
    tb.close()
    del ema

    if full:  # no OOF / val-C; done.json (written last, atomically) marks the run as finished (--skip_done)
        res = {
            "fold": FULL, "select": "last", "ema": bool(e), "selected_epoch": epochs, "epochs": epochs,
            "n_train": len(df_tr), "n_fit": len(df_fit), "final_train_loss": rows[-1]["train_loss"] if rows else None,
            "weights": "state_dict_ema" if e else "state_dict", "seed": seed,
        }
        tmp = out / "done.json.tmp"
        tmp.write_text(json.dumps(res, indent=2))
        tmp.replace(out / "done.json")
        print(f"[full] done: {epochs} epochs on {len(df_tr)} images -> {out / 'last.pt'} ({res['weights']})")
        return res

    # selected checkpoint: EMA weights if EMA is on, else raw; final epoch (last) or best epoch by that AUC (best)
    ck = torch.load(out / ("best.pt" if select == "best" else "last.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict_ema"] if "state_dict_ema" in ck else ck["state_dict"])
    selected_epoch = ck["epoch"]
    del ck
    p, yv = predict(model, val_loader, device)
    pd.DataFrame({"image_id": df_va.image_id, "label": yv.astype(int), "prob": p}).to_csv(out / f"oof_fold{fold}.csv", index=False)
    wtype = "EMA" if e else "raw"
    print(f"\n[fold {fold}] select={select} -> epoch {selected_epoch} ({wtype}); best {wtype} epoch {best_epoch} AUC {best_score:.4f} -> val-C")
    res = full_eval(model, df_va, root, out, eval_bs, workers, args.valc_limit, write_json=False)
    res.update({"select": select, "ema": bool(e), "selected_epoch": selected_epoch, "best_epoch": best_epoch})
    res["pair_breakdown"] = pair_breakdown(df_va.image_id, yv, p)  # PHASE4 §6.2, from the OOF probabilities
    pb = res["pair_breakdown"]
    _f = lambda v: "n/a" if v is None else f"{v:.4f}"
    print(f"[fold {fold}] pair breakdown: AUC pair {_f(pb['auc_pair'])} (n fakes {pb['n_pair_fakes']}) | "
          f"non-pair {_f(pb['auc_nonpair'])} (n fakes {pb['n_nonpair_fakes']}) | reals {pb['n_reals']}")
    # eval.json is written last: its existence marks the fold as done (--skip_done)
    tmp = out / "eval.json.tmp"
    tmp.write_text(json.dumps(res, indent=2))
    tmp.replace(out / "eval.json")  # atomic rename
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
    ap.add_argument("--fold", default="0", help="fold index, comma list (0,1,2) or 'all'")
    ap.add_argument("--data_root")
    ap.add_argument("--run_name", help="default: config file stem")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="subsample train (smoke test)")
    ap.add_argument("--valc_limit", type=int, help="subsample val images for val-C (smoke test)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no_log", action="store_true", help="do not append to EXPERIMENTS.md")
    ap.add_argument("--skip_done", action="store_true", help="reuse folds whose eval.json already exists")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    run = args.run_name or Path(args.config).stem
    run_dir = RUNS_DIR / run
    folds = parse_folds(args.fold)

    results = {}
    for k in folds:
        if args.skip_done and fold_done(run_dir, k):
            if k == FULL:
                results[k] = json.loads((run_dir / FULL / "done.json").read_text())
                print(f"[full] skip: {run_dir / FULL / 'done.json'} exists")
            else:
                done = run_dir / f"fold{k}" / "eval.json"
                results[k] = json.loads(done.read_text())
                print(f"[fold {k}] skip: {done} exists (clean AUC {results[k]['summary']['clean_auc']:.4f})")
            continue
        results[k] = train_fold(cfg, k, root, run_dir, args)
        if k == FULL:
            if not args.no_log:
                r = results[k]
                with open(EXPERIMENTS_MD, "a") as f:
                    f.write(f"| {dt.date.today().isoformat()} | {run}/full | {cfg['model']['name']} | full-data retrain "
                            f"({r['epochs']} ep{', EMA' if r['ema'] else ''}, n={r['n_train']}) | | | | | | "
                            f"final train loss {r['final_train_loss']:.4f}. {args.notes} |\n")
            continue
        if not args.no_log:
            r = results[k]
            note = f"{r['select']} ep {r['selected_epoch']}{' EMA' if r['ema'] else ''} (best ep {r['best_epoch']}). {args.notes}"
            log_experiment(f"{run}/fold{k}", cfg, r["summary"], note.strip())

    if len(results) > 1:
        tag = "".join(str(k) for k in folds)
        oof = pd.concat(pd.read_csv(run_dir / f"fold{k}" / f"oof_fold{k}.csv", dtype={"image_id": str}) for k in folds)
        oof.to_csv(run_dir / f"oof_f{tag}.csv", index=False)
        aucs = [results[k]["summary"]["clean_auc"] for k in folds]
        sels = [results[k]["summary"]["selection_score"] for k in folds]
        msg = (
            f"OOF AUC (folds {','.join(map(str, folds))}, n={len(oof)}): {roc_auc_score(oof.label, oof.prob):.4f} | "
            f"per-fold mean {np.mean(aucs):.4f} +- {np.std(aucs):.4f} | selection mean {np.mean(sels):.4f} +- {np.std(sels):.4f}"
        )
        print("\n" + msg)
        (run_dir / f"oof_summary_f{tag}.txt").write_text(msg + "\n")
        if sorted(folds) == list(range(5)):
            oof.to_csv(run_dir / "oof_all.csv", index=False)
            (run_dir / "oof_summary.txt").write_text(msg + "\n")
            if not args.no_log:
                with open(EXPERIMENTS_MD, "a") as f:
                    f.write(f"| {dt.date.today().isoformat()} | {run}/OOF | {cfg['model']['name']} | 5-fold summary | {msg} | | | | | |\n")

if __name__ == "__main__":
    main()
