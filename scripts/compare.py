"""Paired A/B comparison of training runs (docs/PHASE2.md section 1.2).

Usage:
  .venv/bin/python scripts/compare.py RUN_A RUN_B [RUN_C ...] [--folds 0,1,2]
      [--threshold 0.004] [--veto 0.01] [--md] [--settings]

RUN_A is the reference; every other run is compared to it. Runs are directory
names under outputs/runs/ ($DFD_RUNS_DIR overrides it). Read-only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ["DFD_RUNS_DIR"]) if os.environ.get("DFD_RUNS_DIR") else ROOT / "outputs" / "runs"
COLS = ["clean", "seen", "heldout", "worst", "selection"]


def load(run, fold):
    p = RUNS / run / f"fold{fold}" / "eval.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def metrics(d):
    s = d["summary"]
    return {"clean": s["clean_auc"], "seen": s["valc_seen"], "heldout": s["valc_heldout"],
            "worst": s["worst_auc"], "selection": s["selection_score"],
            "worst_setting": s.get("worst_setting", "?")}


def flat_valc(d):
    return {f"{k}@{sev}": v for k, sevs in d["valc"].items() for sev, v in sevs.items()}


def fmt(x, signed=False):
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def verdict(deltas, a_held, b_held, thr, veto):
    """deltas: dict col -> list over folds. Returns (label, reason)."""
    n = len(deltas["selection"])
    m = {c: sum(v) / n for c, v in deltas.items()}
    sel = deltas["selection"]
    all_pos = all(x > 0 for x in sel)
    veto_hit = m["heldout"] < -veto or m["clean"] < -veto
    if m["selection"] >= thr and all_pos:
        if veto_hit:
            which = []
            if m["heldout"] < -veto:
                which.append(f"mean dheld-out {m['heldout']:+.4f} < -{veto}")
            if m["clean"] < -veto:
                which.append(f"mean dclean {m['clean']:+.4f} < -{veto}")
            return "VETO", "would ADOPT but " + "; ".join(which)
        return "ADOPT", f"mean dselection {m['selection']:+.4f} >= {thr} and positive on all {n} folds"
    if abs(m["selection"]) < thr:
        side = "B" if b_held > a_held else ("A" if a_held > b_held else "neither (equal)")
        return "TIE", (f"|mean dselection| {abs(m['selection']):.4f} < {thr}; keep simpler/cheaper; "
                       f"if equal prefer higher held-out (higher: {side}, A {a_held:.4f} vs B {b_held:.4f})")
    if m["selection"] >= thr:
        return "INCONCLUSIVE", (f"mean dselection {m['selection']:+.4f} >= {thr} but signs disagree "
                                f"({', '.join(fmt(x, True) for x in sel)})")
    return "REJECT", f"mean dselection {m['selection']:+.4f} <= -{thr}" if m["selection"] <= -thr else "no gain"


def render(rows, header, md):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = []
    if md:
        lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
        lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    else:
        lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        lines.append("  ".join("-" * w for w in widths))
        for r in rows:
            lines.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


def compare(a, b, folds, args):
    data = {}
    for run in (a, b):
        for f in folds:
            data[(run, f)] = load(run, f)
    common = [f for f in folds if data[(a, f)] and data[(b, f)]]
    missing = [f for f in folds if f not in common]
    if missing:
        print(f"WARNING: folds {missing} missing in {a} or {b}; comparing on intersection {common}", file=sys.stderr)
    if not common:
        sys.exit(f"ERROR: no common folds between {a} and {b} (requested {folds})")
    if len(common) < 3:
        print(f"WARNING: only {len(common)} fold(s) compared (<3); verdict is weak evidence", file=sys.stderr)
    for key in ("select", "ema"):
        va = {data[(a, f)].get(key) for f in common}
        vb = {data[(b, f)].get(key) for f in common}
        if va != vb:
            print(f"WARNING: '{key}' differs ({a}: {sorted(map(str, va))} vs {b}: {sorted(map(str, vb))}); "
                  f"best-epoch vs last-epoch results are not apples-to-apples", file=sys.stderr)

    header = ["fold", "run", "clean", "seen", "held-out", "worst", "worst setting", "selection"]
    rows, deltas = [], {c: [] for c in COLS}
    for f in common:
        ma, mb = metrics(data[(a, f)]), metrics(data[(b, f)])
        rows.append([f, f"A {a}"] + [fmt(ma[c]) for c in COLS[:4]] + [ma["worst_setting"], fmt(ma["selection"])])
        rows.append(["", f"B {b}"] + [fmt(mb[c]) for c in COLS[:4]] + [mb["worst_setting"], fmt(mb["selection"])])
        d = {c: mb[c] - ma[c] for c in COLS}
        for c in COLS:
            deltas[c].append(d[c])
        rows.append(["", "delta"] + [fmt(d[c], True) for c in COLS[:4]] + ["", fmt(d["selection"], True)])
    n = len(common)
    mean = {c: sum(v) / n for c, v in deltas.items()}
    rows.append(["mean", "delta"] + [fmt(mean[c], True) for c in COLS[:4]] + ["", fmt(mean["selection"], True)])
    # reorder: columns in header are clean, seen, held-out, worst, worst setting, selection
    print(f"\n=== {b}  vs reference {a}  (folds {common}) ===")
    print(render(rows, header, args.md))
    a_h = sum(metrics(data[(a, f)])["heldout"] for f in common) / n
    b_h = sum(metrics(data[(b, f)])["heldout"] for f in common) / n
    label, reason = verdict(deltas, a_h, b_h, args.threshold, args.veto)
    print(f"\nVERDICT: {label}  ({reason})")

    if args.settings:
        acc = {}
        for f in common:
            va, vb = flat_valc(data[(a, f)]), flat_valc(data[(b, f)])
            for k in va.keys() & vb.keys():
                acc.setdefault(k, []).append(vb[k] - va[k])
        ds = sorted(((sum(v) / len(v), k) for k, v in acc.items()), reverse=True)
        k5 = min(5, len(ds))
        srows = [[k, fmt(v, True)] for v, k in ds[:k5]]
        brow = [[k, fmt(v, True)] for v, k in ds[-k5:]] if len(ds) > k5 else []
        print(f"\nPer-setting val-C mean delta ({len(ds)} settings): top {k5}")
        print(render(srows, ["setting", "mean delta"], args.md))
        if brow:
            print(f"\nbottom {k5}")
            print(render(brow, ["setting", "mean delta"], args.md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="RUN_A (reference) RUN_B [RUN_C ...]")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--threshold", type=float, default=0.004)
    ap.add_argument("--veto", type=float, default=0.01)
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    ap.add_argument("--settings", action="store_true", help="per-setting val-C delta, top/bottom 5")
    args = ap.parse_args()
    if len(args.runs) < 2:
        ap.error("need at least RUN_A and RUN_B")
    folds = [int(x) for x in args.folds.split(",") if x != ""]
    a = args.runs[0]
    for r in args.runs:
        if not (RUNS / r).is_dir():
            sys.exit(f"ERROR: run not found: {RUNS / r}")
    for b in args.runs[1:]:
        compare(a, b, folds, args)


if __name__ == "__main__":
    main()
