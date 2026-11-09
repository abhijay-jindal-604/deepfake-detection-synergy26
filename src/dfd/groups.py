"""Pair / near-duplicate discovery on TRAIN images only -> groups -> grouped folds.

Two stages, because the similarity threshold is confirmed by a human in between:

  python -m dfd.groups candidates --data_root <root>      # hashes, embeddings, candidate pairs, audit grids
  python -m dfd.groups folds --threshold 0.82 --data_root <root>   # (0.82 = the value confirmed on the audit grids) union-find groups + StratifiedGroupKFold

Never touches test/images.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import AUDIT_DIR, DATA_DIR, ensure_dirs, resolve_data_root, train_image_path
from .seed import SEED, seed_everything

EMB_CACHE = AUDIT_DIR / "train_embeddings.npz"
CAND_CSV = AUDIT_DIR / "candidate_pairs.csv"
TOPK = 10
PHASH_MAX = 10
BUCKETS = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80)]


def load_meta(root: Path) -> pd.DataFrame:
    meta = pd.read_csv(root / "train_metadata.csv", dtype={"image_id": str})
    meta["path"] = [str(train_image_path(root, i, l)) for i, l in zip(meta.image_id, meta.label)]
    return meta.sort_values("image_id").reset_index(drop=True)


# ---------------------------------------------------------------- hashes
def _hash_one(path: str) -> tuple[int, int]:
    import imagehash
    from PIL import Image

    im = Image.open(path).convert("L")
    return int(str(imagehash.phash(im)), 16), int(str(imagehash.dhash(im)), 16)


def compute_hashes(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_hash_one, paths, chunksize=64))
    ph = np.array([o[0] for o in out], dtype=np.uint64)
    dh = np.array([o[1] for o in out], dtype=np.uint64)
    return ph, dh


def hamming_matrix(h: np.ndarray) -> np.ndarray:
    """All-pairs Hamming distance of 64-bit hashes (7k x 7k uint8)."""
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1)  # N x 64
    b = bits.astype(np.float32)  # float32 -> BLAS; counts <= 64 are exact
    same = b @ b.T + (1 - b) @ (1 - b).T
    return np.rint(64 - same).astype(np.uint8)


# ---------------------------------------------------------------- embeddings
def compute_embeddings(paths: list[str], batch_size: int = 64) -> np.ndarray:
    """ImageNet convnext_tiny, global-pooled, L2-normalised; median-blurred input to damp snow/noise."""
    import cv2
    import timm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0).to(device).eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load(p: str) -> np.ndarray:
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(cv2.medianBlur(img, 3), (224, 224), interpolation=cv2.INTER_AREA)
        return ((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)

    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            x = torch.from_numpy(np.stack([load(p) for p in paths[i : i + batch_size]])).to(device)
            f = model(x).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  embedded {i + len(x)}/{len(paths)}", flush=True)
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------- candidates
def build_candidates(meta: pd.DataFrame, emb: np.ndarray, ph: np.ndarray, dh: np.ndarray) -> pd.DataFrame:
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    hd_p = hamming_matrix(ph)
    hd_d = hamming_matrix(dh)

    pairs: set[tuple[int, int]] = set()
    nn = np.argpartition(-sim, TOPK, axis=1)[:, :TOPK]
    for i, row in enumerate(nn):
        for j in row:
            if sim[i, j] >= BUCKETS[-1][0]:
                pairs.add((min(i, j), max(i, j)))
    ii, jj = np.where(np.triu(hd_p <= PHASH_MAX, k=1))
    pairs.update(zip(ii.tolist(), jj.tolist()))

    rows = []
    for i, j in sorted(pairs):
        rows.append(
            dict(
                id_a=meta.image_id[i], id_b=meta.image_id[j],
                label_a=int(meta.label[i]), label_b=int(meta.label[j]),
                cos_sim=round(float(sim[i, j]), 4),
                phash_dist=int(hd_p[i, j]), dhash_dist=int(hd_d[i, j]),
            )
        )
    df = pd.DataFrame(rows).sort_values("cos_sim", ascending=False).reset_index(drop=True)
    return df


def pair_type(a: int, b: int) -> str:
    return {0: "real-real", 1: "real-fake", 2: "fake-fake"}[a + b]


def cmd_candidates(args) -> None:
    from .viz import pair_grid

    root = resolve_data_root(args.data_root)
    ensure_dirs()
    seed_everything(SEED)
    meta = load_meta(root)
    print(f"{len(meta)} train images ({meta.label.sum()} fake)")

    if EMB_CACHE.exists() and not args.recompute:
        z = np.load(EMB_CACHE, allow_pickle=False)
        assert list(z["ids"]) == list(meta.image_id), "cache does not match metadata"
        emb, ph, dh = z["emb"], z["phash"], z["dhash"]
        print("loaded cached embeddings/hashes")
    else:
        print("hashing ...")
        ph, dh = compute_hashes(meta.path.tolist())
        print("embedding (convnext_tiny, ImageNet) ...")
        emb = compute_embeddings(meta.path.tolist())
        np.savez(EMB_CACHE, ids=meta.image_id.values.astype(str), emb=emb, phash=ph, dhash=dh)

    cand = build_candidates(meta, emb, ph, dh)
    cand["type"] = [pair_type(a, b) for a, b in zip(cand.label_a, cand.label_b)]
    cand.to_csv(CAND_CSV, index=False)
    print(f"\n{len(cand)} candidate pairs -> {CAND_CSV}")

    id2path = dict(zip(meta.image_id, meta.path))
    rng = np.random.default_rng(SEED)
    print(f"\n{'bucket':>12} {'pairs':>6} {'real-fake':>9} {'real-real':>9} {'fake-fake':>9} {'pHash<=10':>9}")
    for lo, hi in BUCKETS:
        b = cand[(cand.cos_sim >= lo) & (cand.cos_sim < hi)]
        tc = b.type.value_counts()
        print(f"{lo:.2f}-{min(hi, 1):.2f} {len(b):>6} {tc.get('real-fake', 0):>9} {tc.get('real-real', 0):>9} "
              f"{tc.get('fake-fake', 0):>9} {(b.phash_dist <= PHASH_MAX).sum():>9}")
        if len(b):
            sample = b.iloc[rng.choice(len(b), size=min(16, len(b)), replace=False)].sort_values("cos_sim", ascending=False)
            out = AUDIT_DIR / f"pairs_{lo:.2f}-{min(hi, 1):.2f}.jpg"
            pair_grid(sample, id2path, out, title=f"cos {lo:.2f}-{min(hi, 1):.2f}  ({len(b)} pairs, 16 random)")
    low = cand[(cand.cos_sim < BUCKETS[-1][0])]
    print(f"{'<0.75 (pHash only)':>12} {len(low):>6}")
    if len(low):
        out = AUDIT_DIR / "pairs_phash_only.jpg"
        sample = low.iloc[rng.choice(len(low), size=min(16, len(low)), replace=False)]
        pair_grid(sample, id2path, out, title=f"pHash<=10 but cos<0.75 ({len(low)} pairs)")
    print(f"\ngrids written to {AUDIT_DIR}")


# ---------------------------------------------------------------- groups + folds
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cmd_folds(args) -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    root = resolve_data_root(args.data_root)
    ensure_dirs()
    meta = load_meta(root)
    cand = pd.read_csv(CAND_CSV, dtype={"id_a": str, "id_b": str})
    keep = cand[(cand.cos_sim >= args.threshold) | (cand.phash_dist <= args.phash_max)].copy()

    idx = {k: i for i, k in enumerate(meta.image_id)}
    uf = UnionFind(len(meta))
    for a, b in zip(keep.id_a, keep.id_b):
        uf.union(idx[a], idx[b])
    roots = np.array([uf.find(i) for i in range(len(meta))])
    _, gid = np.unique(roots, return_inverse=True)
    meta["group_id"] = gid
    meta["group_size"] = meta.groupby("group_id").image_id.transform("size")

    biggest = int(meta.group_size.max())
    print(f"accepted pairs: {len(keep)}  (threshold cos>={args.threshold} or pHash<={args.phash_max})")
    print("by type:", keep.type.value_counts().to_dict())
    print(f"groups: {meta.group_id.nunique()}  | largest group: {biggest}")
    print("group-size histogram:", meta.drop_duplicates("group_id").group_size.value_counts().sort_index().to_dict())
    if biggest > args.max_group:
        raise SystemExit(f"Largest group has {biggest} images (> {args.max_group}); threshold too loose.")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta["fold"] = -1
    for k, (_, va) in enumerate(sgkf.split(meta, meta.label, meta.group_id)):
        meta.loc[va, "fold"] = k
    assert (meta.fold >= 0).all()
    assert meta.groupby("group_id").fold.nunique().max() == 1, "a group spans folds"

    keep[["id_a", "id_b", "label_a", "label_b", "cos_sim", "phash_dist"]].to_csv(DATA_DIR / "pairs.csv", index=False)
    meta[["image_id", "label", "group_id", "group_size"]].to_csv(DATA_DIR / "groups.csv", index=False)
    meta[["image_id", "label", "group_id", "fold"]].to_csv(DATA_DIR / "folds.csv", index=False)

    print("\nfold  n     share  fake_ratio")
    for k, g in meta.groupby("fold"):
        print(f"{k:>4}  {len(g):<5} {len(g) / len(meta):.3f}  {g.label.mean():.3f}")
    print(f"\nwrote {DATA_DIR / 'pairs.csv'}, groups.csv, folds.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--data_root")
    c.add_argument("--recompute", action="store_true")
    f = sub.add_parser("folds")
    f.add_argument("--data_root")
    f.add_argument("--threshold", type=float, required=True, help="cosine similarity to accept a pair")
    f.add_argument("--phash_max", type=int, default=PHASH_MAX)
    f.add_argument("--max_group", type=int, default=50)
    args = p.parse_args()
    {"candidates": cmd_candidates, "folds": cmd_folds}[args.cmd](args)


if __name__ == "__main__":
    main()
