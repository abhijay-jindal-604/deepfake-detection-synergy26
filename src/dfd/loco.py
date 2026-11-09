"""Leave-one-cluster-out (LOCO): our unseen-generation-method proxy (docs/PHASE4.md §6.3).

The TRAINING fakes are clustered into k groups by ImageNet convnext_tiny embeddings (the cached ones from
`dfd.groups candidates`, train images only). For each cluster c a fold-0 run trains on fold-0-train MINUS the
cluster-c fakes (config key `data.loco_holdout: c`), then fold-0-val reals vs fold-0-val cluster-c fakes is the
"unseen method" AUC. Comparing configs (p2_final vs the best SBI config) on that number says which recipe
generalises better to a fake style it never saw.

  # 1. clusters (CPU, seconds) -> data/loco_clusters.csv + outputs/audit/loco_clusters.jpg
  python -m dfd.loco clusters [--k 4] [--emb outputs/audit/train_embeddings.npz]
  # 2. configs + queue for one recipe (fold 0, one run per held-out cluster)
  python -m dfd.loco configs --base configs/p2/p2_final.yaml [--name p2_final]
  #    -> configs/loco/<name>_loco{c}.yaml, queues/loco_<name>.yaml ; then on the GPU:
  python scripts/run_queue.py queues/loco_<name>.yaml
  # 3. report (CPU): held-out vs in-distribution AUC per cluster, for one or more recipes
  python -m dfd.loco report --names p2_final p4_sbi03 [--ref p2_final=p2_final p4_sbi03=p4_sbi03]

Metrics per held-out cluster c (all on fold-0 val, from the LOCO run's oof_fold0.csv):
  auc_heldout   all val reals vs val cluster-c fakes                       (PHASE4 §6.3 definition)
  auc_heldout_m val reals ASSIGNED to cluster c (nearest centroid) vs val cluster-c fakes: content-matched
                variant; it removes the "cluster c is just a different kind of scene" shortcut
  auc_seen      all val reals vs val fakes of the other clusters (sanity: should be ~normal)
  auc_ref       same subset as auc_heldout, scored by the reference run that saw cluster c in training
  gap           auc_ref - auc_heldout (the cost of never having seen this fake style)
Caveat for the report: ImageNet-feature clusters partly reflect scene content, not only the generation
method. `clusters` prints how the train REALS spread over the same centroids; if reals spread like the
fakes, the clusters are mostly content clusters. It is a proxy, not proof.

Never touches test/images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import AUDIT_DIR, DATA_DIR, OUTPUTS_DIR, REPO_ROOT, RUNS_DIR, ensure_dirs
from .seed import SEED

LOCO_CSV = DATA_DIR / "loco_clusters.csv"
LOCO_KEY = "loco_holdout"  # cfg["data"][LOCO_KEY] = held-out cluster id
LOCO_FOLD = 0
DEFAULT_K = 4
MIN_VAL_FAKES = 40  # warn when a cluster has fewer fold-0 val fakes than this (AUC too noisy)


# ---------------------------------------------------------------- clustering
def load_embeddings(emb_path: Path, ids: list[str]) -> np.ndarray:
    z = np.load(emb_path, allow_pickle=False)
    pos = {k: i for i, k in enumerate(z["ids"].astype(str))}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} ids missing from {emb_path} (e.g. {missing[0]}); rerun dfd.groups candidates")
    e = z["emb"][[pos[i] for i in ids]].astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8)


def cluster_fakes(emb_fake: np.ndarray, k: int, seed: int = SEED):
    """KMeans on L2-normalised fake embeddings (cosine geometry). Returns (labels, centroids). Clusters are
    renumbered by descending size so cluster 0 is always the largest (stable ids across reruns)."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb_fake)
    sizes = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[km.labels_], km.cluster_centers_[order]


def nearest_centroid(emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmin(((emb[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)


def cmd_clusters(args) -> None:
    from .data import load_folds

    ensure_dirs()
    folds = load_folds()
    emb = load_embeddings(Path(args.emb), folds.image_id.tolist())
    fake = folds.label.astype(int).to_numpy() == 1
    lab_f, cent = cluster_fakes(emb[fake], args.k, args.seed)
    # every image gets its nearest centroid (fakes: = their KMeans label); reals only for the matched AUC / diagnostic
    near = nearest_centroid(emb, cent)
    near[fake] = lab_f
    out = folds[["image_id", "label", "fold"]].copy()
    out["cluster"] = near
    out["is_fake"] = fake.astype(int)
    out[["image_id", "label", "fold", "cluster"]].to_csv(args.out, index=False)

    k = args.k
    tab = pd.crosstab([out.label.map({0: "real", 1: "fake"}), out.fold == LOCO_FOLD], out.cluster)
    print(f"k={k} clusters on {fake.sum()} train fakes -> {args.out}")
    print("\nimages per cluster (label, is fold-0 val):\n" + tab.to_string())
    f_share = np.bincount(near[fake], minlength=k) / fake.sum()
    r_share = np.bincount(near[~fake], minlength=k) / (~fake).sum()
    print("\nshare of fakes per cluster:", np.round(f_share, 3).tolist())
    print("share of reals (nearest centroid):", np.round(r_share, 3).tolist())
    tv = 0.5 * np.abs(f_share - r_share).sum()
    print(f"total-variation distance fakes vs reals over clusters: {tv:.3f} "
          f"(near 0 = the clusters are mostly scene/content clusters, not fake-specific)")
    val_f = out[(out.fold == LOCO_FOLD) & (out.label == 1)].cluster.value_counts()
    for c in range(k):
        if val_f.get(c, 0) < MIN_VAL_FAKES:
            print(f"WARNING: cluster {c} has only {val_f.get(c, 0)} fold-{LOCO_FOLD} val fakes (< {MIN_VAL_FAKES}); "
                  f"its held-out AUC is noisy. Consider a smaller --k.")
    if not args.no_grid:
        cluster_grid(out, args.data_root, AUDIT_DIR / "loco_clusters.jpg", k, args.seed)


def cluster_grid(df: pd.DataFrame, data_root, out: Path, k: int, seed: int, per: int = 8) -> None:
    """Rows = clusters, 8 random fakes each (audit / report figure)."""
    from PIL import Image

    from .paths import resolve_data_root, train_image_path
    from .viz import LABEL_COLOUR, _tile, grid_from_cells

    try:
        root = resolve_data_root(data_root)
    except SystemExit as e:  # no dataset root -> skip the figure, keep the CSV
        print(f"(no grid: {e})")
        return
    rng = np.random.default_rng(seed)
    cells = []
    for c in range(k):
        ids = df[(df.label == 1) & (df.cluster == c)].image_id.to_numpy()
        pick = rng.choice(ids, size=min(per, len(ids)), replace=False)
        for i in pick:
            cells.append(_tile(Image.open(train_image_path(root, i, 1)), f"cluster {c} (n={len(ids)})",
                               LABEL_COLOUR[1], size=160))
        cells += [Image.new("RGB", cells[-1].size, "white")] * (per - len(pick))
    grid_from_cells(cells, out, f"LOCO: k={k} clusters of TRAIN fakes (convnext_tiny ImageNet features), {per} random each",
                    cols=per, gap=4)
    print(f"grid -> {out}")


# ---------------------------------------------------------------- training hook (used by dfd.train)
def drop_heldout_fakes(df_tr: pd.DataFrame, cluster: int, csv: Path = LOCO_CSV) -> pd.DataFrame:
    """Remove the fakes of `cluster` from a training frame (reals and other fakes untouched)."""
    if not Path(csv).is_file():
        raise FileNotFoundError(f"{csv} missing: run python -m dfd.loco clusters first")
    cl = pd.read_csv(csv, dtype={"image_id": str}).set_index("image_id").cluster
    c = cl.reindex(df_tr.image_id.astype(str))
    if c.isna().any():
        raise ValueError(f"{int(c.isna().sum())} training images missing from {csv}")
    if cluster not in set(cl.unique()):
        raise ValueError(f"held-out cluster {cluster} not in {csv} (clusters {sorted(cl.unique())})")
    drop = (df_tr.label.astype(int).to_numpy() == 1) & (c.to_numpy() == cluster)
    print(f"[loco] hold out cluster {cluster}: dropping {int(drop.sum())} of {int((df_tr.label == 1).sum())} train fakes")
    return df_tr[~drop].reset_index(drop=True)


# ---------------------------------------------------------------- configs + queue
def run_name(name: str, c: int) -> str:
    return f"loco_{name}_c{c}"


def cmd_configs(args) -> None:
    import os

    k = int(pd.read_csv(args.csv).cluster.max()) + 1
    base = Path(args.base).resolve()
    name = args.name or base.stem
    cdir = REPO_ROOT / "configs" / "loco"
    cdir.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(base, cdir)
    runs = []
    for c in range(k):
        p = cdir / f"{name}_loco{c}.yaml"
        p.write_text(
            f"base: {rel}\n"
            f'description: "LOCO (PHASE4 §6.3): {name} with train fakes of cluster {c} held out (fold {LOCO_FOLD} only)"\n'
            f"data: {{{LOCO_KEY}: {c}}}\n"
        )
        runs.append({"config": str(p.relative_to(REPO_ROOT)), "folds": str(LOCO_FOLD), "run_name": run_name(name, c),
                     "timeout_min": args.timeout_min,
                     "args": ["--no_log", "--valc_limit", str(args.valc_limit), "--notes", f"LOCO c{c}"]})
    q = REPO_ROOT / "queues" / f"loco_{name}.yaml"
    q.parent.mkdir(exist_ok=True)
    import yaml

    q.write_text(f"# LOCO (PHASE4 §6.3) for {name}: fold {LOCO_FOLD}, one run per held-out cluster (k={k}).\n"
                 f"# Reference for the gap = the normal {name} fold-{LOCO_FOLD} run. Report: python -m dfd.loco report --names {name}\n"
                 + yaml.safe_dump({"runs": runs}, sort_keys=False))
    print(f"wrote {k} configs in {cdir} and {q}")


# ---------------------------------------------------------------- report
def _auc(y: np.ndarray, p: np.ndarray):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def loco_metrics(oof: pd.DataFrame, cl: pd.Series, c: int, ref: pd.DataFrame | None) -> dict:
    """oof: image_id,label,prob of one LOCO run (fold-0 val); cl: image_id -> cluster."""
    d = oof.assign(cluster=cl.reindex(oof.image_id.astype(str)).to_numpy())
    real, fake = d.label.to_numpy() == 0, d.label.to_numpy() == 1
    inc = d.cluster.to_numpy() == c
    y, p = d.label.to_numpy().astype(int), d.prob.to_numpy()
    held, seen, matched = real | (fake & inc), real | (fake & ~inc), (real & inc) | (fake & inc)
    r = {
        "cluster": c, "n_val_fakes": int((fake & inc).sum()), "n_val_reals_matched": int((real & inc).sum()),
        "auc_heldout": _auc(y[held], p[held]), "auc_heldout_m": _auc(y[matched], p[matched]),
        "auc_seen": _auc(y[seen], p[seen]),
    }
    if ref is not None:
        rp = ref.set_index(ref.image_id.astype(str)).prob.reindex(d.image_id.astype(str)).to_numpy()
        if np.isnan(rp).any():
            raise ValueError("reference OOF does not cover the LOCO run's val images")
        r["auc_ref"] = _auc(y[held], rp[held])
        r["auc_ref_m"] = _auc(y[matched], rp[matched])
        r["gap"] = None if r["auc_ref"] is None or r["auc_heldout"] is None else r["auc_ref"] - r["auc_heldout"]
    return r


def _read_oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str})


def cmd_report(args) -> None:
    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    cl = pd.read_csv(args.csv, dtype={"image_id": str}).set_index("image_id").cluster
    k = int(cl.max()) + 1
    refs = dict(x.split("=", 1) for x in (args.ref or []))
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    out_all, lines = {}, []
    for name in args.names:
        ref_name = refs.get(name, name)
        ref_path = runs_dir / ref_name / f"fold{LOCO_FOLD}" / f"oof_fold{LOCO_FOLD}.csv"
        ref = _read_oof(ref_path) if ref_path.is_file() else None
        if ref is None:
            print(f"[{name}] reference {ref_path} missing: no auc_ref / gap")
        rows = []
        for c in range(k):
            pth = runs_dir / run_name(name, c) / f"fold{LOCO_FOLD}" / f"oof_fold{LOCO_FOLD}.csv"
            if not pth.is_file():
                print(f"[{name}] missing {pth} (run queues/loco_{name}.yaml)")
                continue
            rows.append(loco_metrics(_read_oof(pth), cl, c, ref))
        if not rows:
            continue
        mean = {key: float(np.mean([r[key] for r in rows if r.get(key) is not None]))
                for key in ("auc_heldout", "auc_heldout_m", "auc_seen", "auc_ref", "auc_ref_m", "gap")
                if any(r.get(key) is not None for r in rows)}
        out_all[name] = {"ref": ref_name, "clusters": rows, "mean": mean, "complete": len(rows) == k}
        lines += [f"\n### LOCO: {name} (reference {ref_name}, fold {LOCO_FOLD}, k={k}{'' if len(rows) == k else ', INCOMPLETE'})\n",
                  "| held-out cluster | val fakes | AUC held-out | AUC held-out (content-matched reals) | AUC seen clusters | AUC ref (saw c) | gap |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['cluster']} | {r['n_val_fakes']} | {fmt(r['auc_heldout'])} | {fmt(r['auc_heldout_m'])} | "
                         f"{fmt(r['auc_seen'])} | {fmt(r.get('auc_ref'))} | {fmt(r.get('gap'))} |")
        lines.append(f"| **mean** | | **{fmt(mean.get('auc_heldout'))}** | {fmt(mean.get('auc_heldout_m'))} | "
                     f"{fmt(mean.get('auc_seen'))} | {fmt(mean.get('auc_ref'))} | {fmt(mean.get('gap'))} |")
    if len(out_all) >= 2:
        names = list(out_all)
        a, b = names[0], names[1]
        da = out_all[b]["mean"].get("auc_heldout", np.nan) - out_all[a]["mean"].get("auc_heldout", np.nan)
        lines.append(f"\nΔ mean held-out AUC {b} − {a}: {da:+.4f} (PHASE4 §5 adopts SBI if ≥ +0.01 and selection drop ≤ 0.004)")
    text = "\n".join(lines)
    print(text)
    od = OUTPUTS_DIR / "loco"
    od.mkdir(parents=True, exist_ok=True)
    (od / "loco_report.md").write_text(text + "\n")
    (od / "loco_report.json").write_text(json.dumps(out_all, indent=2))
    print(f"\n-> {od / 'loco_report.md'}, loco_report.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clusters")
    c.add_argument("--k", type=int, default=DEFAULT_K)
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--emb", default=str(AUDIT_DIR / "train_embeddings.npz"), help="cache from dfd.groups candidates")
    c.add_argument("--out", default=str(LOCO_CSV))
    c.add_argument("--data_root", help="for the audit grid only")
    c.add_argument("--no_grid", action="store_true")
    g = sub.add_parser("configs")
    g.add_argument("--base", required=True, help="recipe config, e.g. configs/p2/p2_final.yaml")
    g.add_argument("--name", help="default: base config stem")
    g.add_argument("--csv", default=str(LOCO_CSV))
    g.add_argument("--valc_limit", type=int, default=128, help="LOCO runs need no full val-C")
    g.add_argument("--timeout_min", type=int, default=25)
    r = sub.add_parser("report")
    r.add_argument("--names", nargs="+", required=True, help="recipe names used in `configs` (first = baseline)")
    r.add_argument("--ref", nargs="*", help="name=reference_run (default: the run with the same name)")
    r.add_argument("--csv", default=str(LOCO_CSV))
    r.add_argument("--runs_dir", help="default: $DFD_RUNS_DIR or outputs/runs")
    a = p.parse_args()
    {"clusters": cmd_clusters, "configs": cmd_configs, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
