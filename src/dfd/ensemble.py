"""Ensemble member / weight / temperature / threshold selection from out-of-fold evidence (docs/PHASE5.md §2).

  python -m dfd.ensemble --runs p3_convnexts p3_vitb p3_b4 [--folds 0,1,2,3,4] [--variant plain|tta_flip]
      [--runs_dir <runs>] [--out outputs/ensemble/ensemble.json] [--step 0.05] [--max_members 4]

Inputs per run and fold k: <runs_dir>/<run>/fold{k}/oof_fold{k}.csv (image_id,label,prob) and, if present,
fold{k}/valc_probs.npz (per-image val-C probabilities, 57 settings). --variant tta_flip reads the same files from
fold{k}/tta_flip/ (written by `python -m dfd.evaluate --ckpt .../last.pt --tta_flip --oof`).

Procedure (all on probabilities converted to logits z = log(p / (1 - p)), clipped):
 1. T_m per member: minimise OOF log-loss of sigmoid(z_m / T_m) (bounded scalar search on log T).
 2. Score s = sum_m w_m z_m / T_m. Objective J = 0.5 * pooled OOF AUC + 0.5 * mean val-C AUC over (fold, setting).
    If any run lacks valc_probs for a fold used, J falls back to OOF AUC only for EVERY candidate (warning).
 3. Search: all subsets of size 1..max_members x simplex weight grid (step 0.05, every weight >= step).
 4. Per subset: equal weights unless the tuned weights beat them by >= 0.003 (TUNE_GAIN) AND pass the optimism
    check (weights tuned on the "tune" folds must not lose to equal weights on the "check" folds; default tune 0,1,2
    / check 3,4; with fewer folds: all but the last / the last; one fold: no check -> equal weights).
 5. Best subset by J, then backward pruning: a member stays only if removing it lowers J by >= 0.002 (PRUNE_GAIN).
 6. Diversity: if the result has one architecture family and a >= 2-family subset is within 0.003 of it, take that.
 7. T_ens on the combined OOF score; threshold = F1-max on the ensemble's OOF probabilities (reported with acc).
Output: ensemble.json (+ a markdown table next to it). Runs in seconds to a couple of minutes on CPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

from .paths import OUTPUTS_DIR, RUNS_DIR

TUNE_GAIN = 0.003   # tuned weights must beat equal weights by this much
PRUNE_GAIN = 0.002  # a member must add at least this much J
DIVERSITY_TOL = 0.003
EPS_OOF = 1e-6      # OOF probs are float32
EPS_VALC = 1e-4     # val-C probs are stored as float16 (resolution ~5e-4 near 1)


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def family(timm_name: str) -> str:
    n = timm_name.lower().split(".")[0]
    for key, fam in (("efficientnet", "efficientnet"), ("convnext", "convnext"), ("vit", "vit"), ("deit", "vit"),
                     ("beit", "vit"), ("eva", "vit"), ("swin", "swin"), ("resnet", "resnet"), ("regnet", "regnet"),
                     ("mobilenet", "mobilenet"), ("xception", "xception")):
        if key in n:
            return fam
    return n.split("_")[0]


def fast_auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Tie-aware AUC (Mann-Whitney) along the last axis. y (N,) 0/1; s (..., N)."""
    r = rankdata(s, axis=-1)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[..., pos].sum(axis=-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------
def run_folds(runs_dir: Path, run: str, variant: str) -> list[int]:
    sub = "" if variant == "plain" else variant
    return [k for k in range(5) if (runs_dir / run / f"fold{k}" / sub / f"oof_fold{k}.csv").is_file()]


def run_model_name(runs_dir: Path, run: str) -> str:
    for k in range(5):
        p = runs_dir / run / f"fold{k}" / "config.yaml"
        if p.is_file():
            return yaml.safe_load(p.read_text())["model"]["name"]
    p = runs_dir / run / "full" / "config.yaml"
    return yaml.safe_load(p.read_text())["model"]["name"] if p.is_file() else "unknown"


class Pool:
    """Aligned OOF + val-C logits for the candidate runs on the common folds / images / settings."""

    def __init__(self, runs_dir: Path, runs: list[str], folds: list[int], variant: str = "plain"):
        self.runs, self.folds, self.variant = runs, folds, variant
        sub = "" if variant == "plain" else variant
        self.oof_y, self.oof_z, self.oof_fold, self.oof_ids = [], [], [], []
        self.valc_y, self.valc_z = {}, {}  # fold -> (N,), fold -> (M, S, N)
        self.settings: list[str] = []
        missing_valc = []
        for k in folds:
            dfs = []
            for r in runs:
                d = pd.read_csv(runs_dir / r / f"fold{k}" / sub / f"oof_fold{k}.csv", dtype={"image_id": str})
                dfs.append(d.set_index("image_id"))
            common = sorted(set.intersection(*[set(d.index) for d in dfs]))
            if any(len(d) != len(common) for d in dfs):
                warnings.warn(f"fold {k}: OOF image sets differ between runs; using the {len(common)} common images")
            y = dfs[0].loc[common, "label"].to_numpy().astype(int)
            for d in dfs[1:]:
                assert (d.loc[common, "label"].to_numpy() == y).all(), f"fold {k}: OOF labels disagree"
            self.oof_y.append(y)
            self.oof_z.append(np.stack([logit(d.loc[common, "prob"].to_numpy(), EPS_OOF) for d in dfs]))
            self.oof_fold.append(np.full(len(common), k))
            self.oof_ids += common
            # val-C
            paths = [runs_dir / r / f"fold{k}" / sub / "valc_probs.npz" for r in runs]
            absent = [r for r, p in zip(runs, paths) if not p.is_file()]
            if absent:
                missing_valc += [f"{r}/fold{k}" for r in absent]
                continue
            zs = [np.load(p) for p in paths]
            ids = [list(z["image_id"].astype(str)) for z in zs]
            common_v = sorted(set.intersection(*[set(i) for i in ids]))
            sets = [set(f for f in z.files if f not in ("image_id", "label")) for z in zs]
            settings = sorted(set.intersection(*sets))
            if self.settings and settings != self.settings:
                settings = sorted(set(settings) & set(self.settings))
            self.settings = settings
            arr, yv = [], None
            for z, i in zip(zs, ids):
                pos = {iid: j for j, iid in enumerate(i)}
                idx = np.array([pos[c] for c in common_v])
                lab = z["label"][idx].astype(int)
                assert yv is None or (lab == yv).all(), f"fold {k}: val-C labels disagree"
                yv = lab
                arr.append(np.stack([logit(z[s][idx].astype(np.float64), EPS_VALC) for s in settings]))
            self.valc_y[k] = yv
            self.valc_z[k] = np.stack(arr)  # (M, S, N)
        # keep only settings common to every fold
        if self.valc_z:
            for k in list(self.valc_z):
                if self.valc_z[k].shape[1] != len(self.settings):
                    raise RuntimeError("val-C settings differ across folds")
        self.oof_y = np.concatenate(self.oof_y)
        self.oof_z = np.concatenate(self.oof_z, axis=1)  # (M, N)
        self.oof_fold = np.concatenate(self.oof_fold)
        self.has_valc = not missing_valc
        if missing_valc:
            warnings.warn("valc_probs.npz missing for " + ", ".join(missing_valc) +
                          ": objective falls back to OOF AUC only (clean-only) for all candidates")
            self.valc_y, self.valc_z = {}, {}

    # ---- objective -------------------------------------------------------------------------------------------
    def score_oof(self, idx: tuple[int, ...], w: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.tensordot(w / t[list(idx)], self.oof_z[list(idx)], axes=1)

    def objective(self, idx: tuple[int, ...], w: np.ndarray, t: np.ndarray, folds: list[int] | None = None) -> dict:
        folds = self.folds if folds is None else folds
        coef = w / t[list(idx)]
        sel = np.isin(self.oof_fold, folds)
        s = coef @ self.oof_z[list(idx)][:, sel]
        auc = float(fast_auc(self.oof_y[sel], s))
        out = {"oof_auc": auc}
        if self.has_valc:
            per = []
            for k in folds:
                sv = np.tensordot(coef, self.valc_z[k][list(idx)], axes=1)  # (S, N)
                per.append(fast_auc(self.valc_y[k], sv))
            vm = float(np.mean(np.concatenate(per)))
            out["valc_auc"] = vm
            out["J"] = 0.5 * auc + 0.5 * vm
        else:
            out["J"] = auc
        return out


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    def nll(logt):
        q = z / math.exp(logt)
        return float(np.mean(np.logaddexp(0, q) - y * q))  # BCE with logits

    r = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded", options={"xatol": 1e-5})
    return float(math.exp(r.x))


def simplex_grid(k: int, step: float) -> np.ndarray:
    n = int(round(1 / step))
    if k == 1:
        return np.ones((1, 1))
    pts = [c for c in itertools.product(range(1, n), repeat=k - 1) if sum(c) < n]
    return np.array([[*c, n - sum(c)] for c in pts], dtype=np.float64) / n


def split_folds(folds: list[int]) -> tuple[list[int], list[int]]:
    if set(range(5)) <= set(folds):
        return [0, 1, 2], [3, 4]
    if len(folds) >= 2:
        return folds[:-1], folds[-1:]
    return folds, []


# ---------------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------------
def threshold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    b = int(np.argmax(f1s))
    t = float(thr[b])
    return {"threshold": t, "f1": float(f1_score(y, p >= t)), "acc": float(accuracy_score(y, p >= t)),
            "f1@0.5": float(f1_score(y, p >= 0.5)), "acc@0.5": float(accuracy_score(y, p >= 0.5)),
            "pred_fake_rate": float((p >= t).mean())}


def select(pool: Pool, families: list[str], step: float = 0.05, max_members: int = 4, log=print) -> dict:
    M = len(pool.runs)
    temps = np.array([fit_temperature(pool.oof_z[m], pool.oof_y) for m in range(M)])
    tune_f, check_f = split_folds(pool.folds)
    cands = {}
    for size in range(1, min(max_members, M) + 1):
        grid = simplex_grid(size, step)
        for idx in itertools.combinations(range(M), size):
            eq = np.full(size, 1.0 / size)
            j_eq = pool.objective(idx, eq, temps)
            c = {"idx": idx, "eq": j_eq, "weights": eq, "tuned": None, "use": "equal", "J": j_eq["J"]}
            if size > 1:
                js = [pool.objective(idx, w, temps)["J"] for w in grid]
                b = int(np.argmax(js))
                w_all = grid[b]
                c["tuned"] = {"weights": w_all.tolist(), "J": float(js[b])}
                if js[b] - j_eq["J"] >= TUNE_GAIN:
                    if check_f:
                        jt = [pool.objective(idx, w, temps, tune_f)["J"] for w in grid]
                        w_t = grid[int(np.argmax(jt))]
                        chk_t = pool.objective(idx, w_t, temps, check_f)["J"]
                        chk_e = pool.objective(idx, eq, temps, check_f)["J"]
                        c["optimism"] = {"tune_folds": tune_f, "check_folds": check_f, "weights_tuned_on_tune": w_t.tolist(),
                                         "J_check_tuned": chk_t, "J_check_equal": chk_e, "passed": bool(chk_t >= chk_e)}
                        if chk_t >= chk_e:
                            c.update(use="tuned", weights=w_all, J=float(js[b]))
                    else:
                        c["optimism"] = {"passed": False, "note": "single fold: no optimism check, equal weights"}
            cands[idx] = c
    best = max(cands.values(), key=lambda c: c["J"])
    log(f"best subset by J: {[pool.runs[i] for i in best['idx']]} J {best['J']:.4f} ({best['use']} weights)")
    # backward pruning
    cur = best
    while len(cur["idx"]) > 1:
        drops = [(cur["J"] - cands[tuple(i for i in cur["idx"] if i != m)]["J"], m) for m in cur["idx"]]
        loss, m = min(drops)
        if loss >= PRUNE_GAIN:
            break
        log(f"prune {pool.runs[m]}: removing it costs only {loss:+.4f} J (< {PRUNE_GAIN})")
        cur = cands[tuple(i for i in cur["idx"] if i != m)]
    # diversity
    fams = {families[i] for i in cur["idx"]}
    diversity_note = f"{len(fams)} family(ies): {sorted(fams)}"
    if len(fams) < 2:
        multi = [c for c in cands.values() if len({families[i] for i in c["idx"]}) >= 2]
        if multi:
            bm = max(multi, key=lambda c: c["J"])
            if bm["J"] >= cur["J"] - DIVERSITY_TOL:
                diversity_note += (f" -> switched to 2-family subset {[pool.runs[i] for i in bm['idx']]} "
                                   f"(J {bm['J']:.4f} vs {cur['J']:.4f}, within {DIVERSITY_TOL})")
                cur = bm
            else:
                diversity_note += f"; best >=2-family subset J {bm['J']:.4f} is > {DIVERSITY_TOL} below: kept"
        else:
            diversity_note += "; no >=2-family candidate in the pool"
    log("diversity: " + diversity_note)
    return {"temps": temps, "cands": cands, "chosen": cur, "best_raw": best, "diversity": diversity_note,
            "tune_folds": tune_f, "check_folds": check_f}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="candidate run names (<= 4 recommended)")
    ap.add_argument("--folds", help="comma list; default: folds present for every run")
    ap.add_argument("--variant", default="plain", choices=["plain", "tta_flip"])
    ap.add_argument("--runs_dir", help="default: $DFD_RUNS_DIR or outputs/runs")
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "ensemble" / "ensemble.json"))
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_members", type=int, default=4)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else RUNS_DIR
    if len(set(args.runs)) != len(args.runs):
        raise SystemExit("duplicate run names")
    avail = [set(run_folds(runs_dir, r, args.variant)) for r in args.runs]
    for r, a in zip(args.runs, avail):
        if not a:
            raise SystemExit(f"{runs_dir / r}: no fold{{k}}/{'' if args.variant == 'plain' else args.variant + '/'}oof_fold{{k}}.csv")
    common = sorted(set.intersection(*avail))
    folds = [int(x) for x in args.folds.split(",")] if args.folds else common
    if not set(folds) <= set(common):
        raise SystemExit(f"folds {folds} not available for every run (common: {common})")
    if not folds:
        raise SystemExit("no fold common to all runs")
    names = [run_model_name(runs_dir, r) for r in args.runs]
    fams = [family(n) for n in names]
    print(f"runs: {', '.join(f'{r} ({n}, {f})' for r, n, f in zip(args.runs, names, fams))} | folds {folds} | variant {args.variant}")

    pool = Pool(runs_dir, args.runs, folds, args.variant)
    print(f"OOF images {len(pool.oof_y)} | val-C: {'%d settings' % len(pool.settings) if pool.has_valc else 'none (clean-only J)'}")
    res = select(pool, fams, args.step, args.max_members)
    temps = res["temps"]
    ch = res["chosen"]
    idx, w = ch["idx"], np.asarray(ch["weights"], dtype=np.float64)

    # final ensemble temperature, threshold, metrics
    s = pool.score_oof(idx, w, temps)
    t_ens = fit_temperature(s, pool.oof_y)
    p = sigmoid(s / t_ens)
    thr = threshold_metrics(pool.oof_y, p)
    per_fold = {}
    for k in folds:
        sel = pool.oof_fold == k
        yk, pk = pool.oof_y[sel], p[sel]
        per_fold[str(k)] = {"auc": float(roc_auc_score(yk, pk)), "f1@thr": float(f1_score(yk, pk >= thr["threshold"])),
                            "acc@thr": float(accuracy_score(yk, pk >= thr["threshold"]))}
    j = pool.objective(idx, w, temps)

    singles = {}
    for m, r in enumerate(args.runs):
        jm = pool.objective((m,), np.ones(1), temps)
        singles[r] = {**jm, "temperature": float(temps[m]), "family": fams[m], "timm_name": names[m]}
    top = sorted(res["cands"].values(), key=lambda c: -c["J"])[:15]
    table = [{"members": [args.runs[i] for i in c["idx"]], "J": c["J"], "use": c["use"],
              "weights": np.asarray(c["weights"]).round(3).tolist(), "J_equal": c["eq"]["J"],
              "oof_auc_equal": c["eq"]["oof_auc"], "valc_auc_equal": c["eq"].get("valc_auc"),
              "J_tuned": (c["tuned"] or {}).get("J"), "optimism": c.get("optimism")} for c in top]

    out = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "runs_dir": str(runs_dir), "variant": args.variant, "folds": folds,
        "objective": "J = 0.5*pooled OOF AUC + 0.5*mean val-C AUC (fold x setting)" if pool.has_valc else
                     "J = pooled OOF AUC (val-C missing for some run: clean-only)",
        "n_oof": int(len(pool.oof_y)), "n_valc_settings": len(pool.settings) if pool.has_valc else 0,
        "members": [{"run": args.runs[i], "timm_name": names[i], "family": fams[i], "temperature": float(temps[i]),
                     "weight": float(wi)} for i, wi in zip(idx, w)],
        "weights_mode": ch["use"],
        "temperature_ens": float(t_ens),
        "threshold": thr["threshold"],
        "metrics": {**j, "oof_threshold": thr, "per_fold": per_fold},
        "optimism_check": ch.get("optimism"),
        "best_before_pruning": [args.runs[i] for i in res["best_raw"]["idx"]],
        "diversity": res["diversity"],
        "rules": {"tune_gain": TUNE_GAIN, "prune_gain": PRUNE_GAIN, "diversity_tol": DIVERSITY_TOL, "step": args.step,
                  "tune_folds": res["tune_folds"], "check_folds": res["check_folds"]},
        "singles": singles,
        "top_candidates": table,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))

    md = ["| members | weights | J | OOF AUC (eq) | val-C AUC (eq) | J tuned | mode |", "|---|---|---|---|---|---|---|"]
    for r in table:
        v = "n/a" if r["valc_auc_equal"] is None else "%.4f" % r["valc_auc_equal"]
        jt = "" if r["J_tuned"] is None else "%.4f" % r["J_tuned"]
        md.append(f"| {' + '.join(r['members'])} | {r['weights']} | {r['J']:.4f} | {r['oof_auc_equal']:.4f} | "
                  f"{v} | {jt} | {r['use']} |")
    outp.with_suffix(".md").write_text("\n".join(md) + "\n")

    print("\nsingle members:")
    for r, sm in singles.items():
        v = "n/a" if sm.get("valc_auc") is None else "%.4f" % sm["valc_auc"]
        print(f"  {r:<24} T {sm['temperature']:.3f} | OOF AUC {sm['oof_auc']:.4f} | val-C {v} | J {sm['J']:.4f}")
    print("\ntop candidates:\n" + "\n".join(md[:12]))
    print(f"\nCHOSEN: {[(m['run'], round(m['weight'], 3), round(m['temperature'], 3)) for m in out['members']]} "
          f"({ch['use']} weights) | J {j['J']:.4f} | OOF AUC {j['oof_auc']:.4f}"
          + (f" | val-C {j['valc_auc']:.4f}" if 'valc_auc' in j else "")
          + f"\nT_ens {t_ens:.3f} | threshold {thr['threshold']:.4f} -> OOF F1 {thr['f1']:.4f} acc {thr['acc']:.4f} "
          f"(@0.5: F1 {thr['f1@0.5']:.4f} acc {thr['acc@0.5']:.4f}) | fold-0 " +
          (f"F1 {per_fold['0']['f1@thr']:.4f} acc {per_fold['0']['acc@thr']:.4f}" if "0" in per_fold else "n/a"))
    print(f"wrote {outp} (+ {outp.with_suffix('.md').name})")


if __name__ == "__main__":
    main()
