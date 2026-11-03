"""Validate a predictions.csv against the hackathon submission format.

Usage:
  python scripts/validate_predictions.py predictions.csv [--test_dir DIR | --ids_csv sample_submission.csv]
         [--compare other_predictions.csv] [--pred PATH] [--threshold T] [--tol 1e-6]

--test_dir only lists file names (os.listdir); images are never opened.
Exit code 1 if any check FAILs.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png")
HEADER = "image_id,label,confidence"
FAILS = 0


def check(ok, name, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def read_raw(path):
    """Read as strings so we can judge formatting ourselves."""
    with open(path, "rb") as f:
        raw = f.read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return df, first.strip("\r\n"), bom


def expected_ids(args):
    if args.test_dir:
        names = os.listdir(args.test_dir)
        return {os.path.splitext(n)[0] for n in names if n.lower().endswith(IMG_EXT)}, f"test_dir ({len(names)} entries listed)"
    if args.ids_csv:
        d = pd.read_csv(args.ids_csv, dtype=str, keep_default_na=False)
        return set(d["image_id"]), "ids_csv"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="?")
    ap.add_argument("--pred", dest="pred")
    ap.add_argument("--threshold", type=float, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--test_dir")
    g.add_argument("--ids_csv")
    ap.add_argument("--compare")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    if args.pred and args.predictions and args.pred != args.predictions:
        ap.error("give the predictions path either positionally or via --pred, not both")
    args.predictions = args.pred or args.predictions
    if not args.predictions:
        ap.error("predictions path required (positional or --pred)")

    df, first, bom = read_raw(args.predictions)
    check(not bom, "no BOM")
    check(first == HEADER, "exact header", f"got {first!r}")
    check(list(df.columns) == HEADER.split(","), "columns exactly image_id,label,confidence", str(list(df.columns)))
    if list(df.columns) != HEADER.split(","):
        print("Cannot continue: wrong columns.")
        return 1

    n = len(df)
    exp, src = expected_ids(args)
    if exp is not None:
        check(n == len(exp), "row count", f"{n} rows vs {len(exp)} expected from {src}")
    else:
        print(f"[INFO] rows: {n} (no --test_dir/--ids_csv given; ID-set check skipped)")

    empty = (df == "").any(axis=1) | df.apply(lambda c: c.str.strip().str.lower().isin(["nan", "none", "null"])).any(axis=1)
    check(not empty.any(), "no NaN/empty cells", f"{int(empty.sum())} bad rows")
    check(not df["image_id"].duplicated().any(), "no duplicate image_ids", f"{int(df['image_id'].duplicated().sum())} dups")
    suffix = df["image_id"].str.lower().str.endswith(IMG_EXT)
    check(not suffix.any(), "image_id has no image extension", f"{int(suffix.sum())} with extension")
    ws = (df["image_id"] != df["image_id"].str.strip())
    check(not ws.any(), "image_id has no surrounding whitespace", f"{int(ws.sum())}")

    if exp is not None:
        got = set(df["image_id"])
        missing, extra = exp - got, got - exp
        check(not missing and not extra, "ID set equals expected", f"missing {len(missing)}, extra {len(extra)}")
        for lbl, s in (("missing", missing), ("extra", extra)):
            if s:
                print(f"       {lbl} e.g. {sorted(s)[:3]}")

    # label
    lab = pd.to_numeric(df["label"], errors="coerce")
    lab_ok = lab.notna() & lab.isin([0, 1]) & df["label"].str.fullmatch(r"[01]")
    check(lab_ok.all(), "label is int in {0,1}", f"{int((~lab_ok).sum())} bad")
    conf = pd.to_numeric(df["confidence"], errors="coerce")
    finite = np.isfinite(conf)
    check(finite.all(), "confidence numeric & finite", f"{int((~finite).sum())} bad")
    inr = finite & (conf >= 0) & (conf <= 1)
    check(inr.all(), "confidence in [0,1]", f"{int((~inr).sum())} out of range")

    if lab_ok.all() and finite.all():
        c0 = conf[lab == 0]
        c1 = conf[lab == 1]
        hi0 = c0.max() if len(c0) else -np.inf
        lo1 = c1.min() if len(c1) else np.inf
        ok = hi0 < lo1 or hi0 == -np.inf or lo1 == np.inf
        check(ok, "labels consistent with a single threshold",
              f"max conf(label0)={hi0:.6g}, min conf(label1)={lo1:.6g}; implied threshold in ({hi0:.6g}, {lo1:.6g}]")
        if not ok:
            bad = int(((lab == 0) & (conf >= lo1)).sum() + ((lab == 1) & (conf <= hi0)).sum())
            print(f"       overlapping rows (approx): {bad}")
        if args.threshold is not None:
            t = args.threshold
            mm = int((lab != (conf >= t).astype(int)).sum())
            check(mm == 0, f"label == (confidence >= {t:g}) for every row", f"{mm} mismatches")
            near = int(((conf - t).abs() <= 0.02).sum())
            print(f"[INFO] confidences within +-0.02 of {t:g}: {near} ({near / max(n, 1):.4f})")
        print(f"[INFO] fraction predicted fake: {float((lab == 1).mean()):.4f}")
    if finite.any():
        q = conf[finite].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        print("[INFO] confidence quantiles: " + ", ".join(f"q{k:g}={v:.4g}" for k, v in q.items()))

    if args.compare:
        o, first2, _ = read_raw(args.compare)
        same_cols = list(o.columns) == HEADER.split(",")
        check(same_cols, "compare: other file has same columns")
        if same_cols:
            check(set(o["image_id"]) == set(df["image_id"]) and len(o) == len(df) and not o["image_id"].duplicated().any(),
                  "compare: same IDs")
            m = df.merge(o, on="image_id", suffixes=("_a", "_b"))
            ca = pd.to_numeric(m["confidence_a"], errors="coerce")
            cb = pd.to_numeric(m["confidence_b"], errors="coerce")
            d = (ca - cb).abs()
            mx = float(d.max()) if len(d) else float("nan")
            check(bool(mx <= args.tol), f"compare: max |dconf| <= {args.tol:g}", f"max={mx:.3g}")
            mism = int((m["label_a"] != m["label_b"]).sum())
            check(mism == 0, "compare: label mismatches == 0", f"{mism} mismatches")

    print("\nRESULT: " + ("ALL PASS" if FAILS == 0 else f"{FAILS} CHECK(S) FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
