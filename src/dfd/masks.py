"""Pair-difference masks (PHASE4 §1): "where is the swap" labels from real/fake pairs of the same photo.

  python -m dfd.masks [--data_root <root>]

For every real-fake pair in data/pairs.csv (same-label pairs ignored):
  1. grey + Gaussian blur σ=2 on both images;
  2. cv2.phaseCorrelate for a translation; reject if |shift| > MAX_SHIFT px or response < MIN_RESPONSE;
     the real is shifted into the fake's frame before differencing;
  3. d = |fake - real| blurred, Otsu threshold, open 5x5, close 11x11, largest component, fill holes,
     feather (blur 7) -> soft mask in [0, 1];
  4. reject if mask area outside [MIN_AREA, MAX_AREA] or mean d outside the mask > MAX_OUTSIDE_D
     (the two images got different corruptions);
  5. a fake with several real partners keeps the accepted pair with the highest cos_sim.

Outputs: data/masks/<fake_id>.png (8-bit, 224x224, fake frame), data/masks_meta.csv (one row per
real-fake pair evaluated), data/mask_prior.npy (mean accepted mask, float32), outputs/audit/pair_masks.jpg.
Training images only.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .paths import AUDIT_DIR, DATA_DIR, ensure_dirs, resolve_data_root, train_image_path
from .seed import SEED
from .viz import _font, grid_from_cells

SIZE = 224
BLUR_SIGMA = 2.0        # step 1 pre-blur
MAX_SHIFT = 8.0         # px, step 2
MIN_RESPONSE = 0.3      # phaseCorrelate peak, step 2
DIFF_BLUR_SIGMA = 2.0   # step 3 blur of |d|
OPEN_K, CLOSE_K, FEATHER_K = 5, 11, 7
MIN_AREA, MAX_AREA = 0.02, 0.45   # fraction of the image, step 4
MAX_OUTSIDE_D = 8.5     # grey levels (0-255) after gain/offset match; mean |d| outside the mask, step 4.
                        # First audit (4 Nov): weak on its own; good masks reach ~8, so it only drops the tail.
# deviation from §1 (see pair_mask): affine ECC alignment, refined from the phaseCorrelate shift
ALIGN = "ecc"           # "phase" = §1 exactly (translation only)
MIN_ECC_CC = 0.90       # ECC correlation of the blurred greys (whole frame, swapped face included); below = "align"
MIN_QUALITY_CC = 0.97   # quality gate: 0.90-0.97 aligns but the pair mostly differs by corruption -> "low_cc"
                        # (first audit: 12/14 visually good masks at cc >= 0.98, mostly bad at 0.93-0.96)
SCALE_RANGE = (0.70, 1.30)  # per-axis (singular values of the affine): some fakes are re-cropped with a new
MAX_ANISO = 1.35            # aspect ratio (e.g. x1.03 / y0.83), so anisotropy is real; ratio above = "geometry"
MAX_SHIFT_ECC = 64.0    # px, centre displacement after the scale
MIN_COVERAGE = 0.60     # fraction of the fake frame with valid (in-frame, non-flat) real pixels
FLAT_K, FLAT_STD = 9, 1.0  # constant-colour block detector (cutout / patch corruptions)

MASK_DIR = DATA_DIR / "masks"
META_CSV = DATA_DIR / "masks_meta.csv"
PRIOR_NPY = DATA_DIR / "mask_prior.npy"
AUDIT_JPG = AUDIT_DIR / "pair_masks.jpg"


def _read_grey(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _flat(raw: np.ndarray) -> np.ndarray:
    """Pixels inside constant-colour blocks (cutout / block-patch corruptions): local std < FLAT_STD."""
    k = (FLAT_K, FLAT_K)
    var = cv2.blur(raw * raw, k) - cv2.blur(raw, k) ** 2
    flat = (np.sqrt(np.maximum(var, 0)) < FLAT_STD).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((FLAT_K, FLAT_K), np.uint8))  # only real blocks
    return cv2.dilate(flat, np.ones((5, 5), np.uint8)).astype(bool)


def _ecc(f: np.ndarray, r: np.ndarray, init: np.ndarray) -> tuple[float, np.ndarray]:
    """Affine ECC, coarse (112) then fine (224). Returns (cc, W) with r(W x) ≈ f(x); cc=-1 on failure."""
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    W = init.astype(np.float32).copy()
    W[:, 2] /= 2
    try:
        _, W = cv2.findTransformECC(cv2.resize(f, (SIZE // 2, SIZE // 2)), cv2.resize(r, (SIZE // 2, SIZE // 2)),
                                    W, cv2.MOTION_AFFINE, crit, None, 5)
        W[:, 2] *= 2
        cc, W = cv2.findTransformECC(f, r, W, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return -1.0, init
    return float(cc), W


def _largest_cc_filled(binary: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cc = (lab == k).astype(np.uint8)
    # fill holes: flood the background from outside (pad so every border pixel connects)
    pad = np.pad(cc, 1)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (cc | holes).astype(np.uint8)


def pair_mask(fake_path, real_path, align: str = ALIGN) -> dict:
    """Soft mask in the fake's frame plus diagnostics (mask is None if alignment is rejected).

    align="phase": PHASE4 §1 as written (translation only, phaseCorrelate).
    align="ecc":   phaseCorrelate result (if plausible) initialises an affine ECC; needed because most
                   fakes are a re-crop of the real (zoom 0.8-1.25, shift up to ~40 px), see report.
    """
    fr, rr = _read_grey(fake_path), _read_grey(real_path)
    f = cv2.GaussianBlur(fr, (0, 0), BLUR_SIGMA)
    r = cv2.GaussianBlur(rr, (0, 0), BLUR_SIGMA)
    win = cv2.createHanningWindow((SIZE, SIZE), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(f, r, win)   # r ≈ f shifted by (dx, dy)
    out = dict(response=round(float(resp), 3), ecc_cc=np.nan, scale=1.0, shift_x=round(float(dx), 2),
               shift_y=round(float(dy), 2), coverage=1.0, area_frac=np.nan, outside_mean_d=np.nan,
               inside_mean_d=np.nan, mask=None, reason="")
    phase_ok = np.hypot(dx, dy) <= MAX_SHIFT and resp >= MIN_RESPONSE
    W = np.float32([[1, 0, dx], [0, 1, dy]])
    if align == "phase":
        if np.hypot(dx, dy) > MAX_SHIFT:
            out["reason"] = "shift"
            return out
        if resp < MIN_RESPONSE:
            out["reason"] = "response"
            return out
    else:
        cc, W = _ecc(f, r, W if phase_ok else np.eye(2, 3, dtype=np.float32))
        A = W[:, :2]
        sc = float(np.sqrt(abs(np.linalg.det(A)))) if cc > 0 else np.nan
        c = np.float32([SIZE / 2, SIZE / 2])
        t = A @ c + W[:, 2] - c                         # where the fake's centre lands in the real
        out.update(ecc_cc=round(cc, 4), scale=round(sc, 4), shift_x=round(float(t[0]), 2),
                   shift_y=round(float(t[1]), 2))
        # affine must be close to a similarity (no shear / anisotropy)
        sv = np.linalg.svd(A, compute_uv=False) if cc > 0 else np.array([1.0, 0.0])
        if cc < MIN_ECC_CC:
            out["reason"] = "align"
            return out
        if not (SCALE_RANGE[0] <= sv.min() and sv.max() <= SCALE_RANGE[1]) or sv[0] / max(sv[1], 1e-6) > MAX_ANISO \
                or np.hypot(*t) > MAX_SHIFT_ECC:
            out["reason"] = "geometry"
            return out
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    r_al = cv2.warpAffine(r, W, (SIZE, SIZE), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones((SIZE, SIZE), np.uint8), W, (SIZE, SIZE), flags=cv2.INTER_NEAREST
                           + cv2.WARP_INVERSE_MAP, borderValue=0).astype(bool)
    flat_r = cv2.warpAffine(_flat(rr).astype(np.uint8), W, (SIZE, SIZE), flags=cv2.INTER_NEAREST
                            + cv2.WARP_INVERSE_MAP, borderValue=0).astype(bool)
    valid &= ~(_flat(fr) | flat_r)                   # ignore cutout / block patches in either image
    out["coverage"] = round(float(valid.mean()), 3)
    if valid.mean() < MIN_COVERAGE:
        out["reason"] = "coverage"
        return out
    # photometric match (gain/offset of the real onto the fake), robust: fit, drop top 40% residuals, refit
    x, y = r_al[valid], f[valid]
    keep = np.ones_like(x, bool)
    for _ in range(2):
        g, o = np.polyfit(x[keep], y[keep], 1)
        res = np.abs(g * x + o - y)
        keep = res <= np.percentile(res, 60)
    d = np.abs(f - (g * r_al + o))
    d[~valid] = 0
    d = cv2.GaussianBlur(d, (0, 0), DIFF_BLUR_SIGMA)
    d8 = np.clip(d, 0, 255).astype(np.uint8)
    _, b = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K)))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K)))
    hard = _largest_cc_filled(b)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (FEATHER_K, FEATHER_K), 0)
    area = float(hard.mean())
    outside, inside = valid & (hard == 0), hard == 1
    out.update(area_frac=round(area, 4), mask=soft,
               outside_mean_d=round(float(d[outside].mean()), 3) if outside.any() else np.nan,
               inside_mean_d=round(float(d[inside].mean()), 3) if inside.any() else np.nan)
    return out


def _reason(row) -> str:
    if row["reason"]:
        return row["reason"]
    if row["ecc_cc"] == row["ecc_cc"] and row["ecc_cc"] < MIN_QUALITY_CC:   # NaN in phase mode
        return "low_cc"
    if row["area_frac"] < MIN_AREA:
        return "area_small"
    if row["area_frac"] > MAX_AREA:
        return "area_large"
    if row["outside_mean_d"] > MAX_OUTSIDE_D:
        return "outside_d"
    return ""


def real_fake_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    m = pairs[pairs.label_a != pairs.label_b]
    a_fake = m.label_a == 1
    return pd.DataFrame({
        "fake_id": np.where(a_fake, m.id_a, m.id_b),
        "real_id": np.where(a_fake, m.id_b, m.id_a),
        "cos_sim": m.cos_sim.values, "phash_dist": m.phash_dist.values,
    }).reset_index(drop=True)


def build_pair_masks(pairs: pd.DataFrame, root, align: str = ALIGN) -> tuple[pd.DataFrame, dict]:
    folds = pd.read_csv(DATA_DIR / "folds.csv").set_index("image_id")
    rf = real_fake_pairs(pairs)
    rows, masks = [], {}
    for k, p in enumerate(rf.itertuples()):
        fa, fr = folds.loc[p.fake_id], folds.loc[p.real_id]
        assert fa.group_id == fr.group_id and fa.fold == fr.fold, (p.fake_id, p.real_id)
        res = pair_mask(train_image_path(root, p.fake_id, 1), train_image_path(root, p.real_id, 0), align)
        masks[k] = res.pop("mask")
        rows.append(dict(fake_id=p.fake_id, real_id=p.real_id, fold_fake=int(fa.fold), fold_real=int(fr.fold),
                         group_id=int(fa.group_id), cos_sim=p.cos_sim, **res))
    meta = pd.DataFrame(rows)
    meta["reject_reason"] = [_reason(r) for r in meta.to_dict("records")]
    meta = meta.drop(columns="reason")
    # step 5: one mask per fake, from the passing partner with the highest cos_sim
    ok = meta[meta.reject_reason == ""].sort_values("cos_sim", ascending=False)
    dup = ok.index[ok.duplicated("fake_id", keep="first")]
    meta.loc[dup, "reject_reason"] = "dup_partner"
    meta["accepted"] = (meta.reject_reason == "").astype(int)
    return meta, masks


def audit_grid(meta: pd.DataFrame, masks: dict, root, n_acc: int = 32, n_rej: int = 16, tile: int = 128) -> None:
    rng = np.random.default_rng(SEED)
    acc = meta.index[meta.accepted == 1].to_numpy()
    rej = meta.index[(meta.accepted == 0) & (meta.reject_reason != "dup_partner")].to_numpy()
    pick = list(rng.choice(acc, min(n_acc, len(acc)), replace=False)) + \
        list(rng.choice(rej, min(n_rej, len(rej)), replace=False))
    bar, cells = 20, []
    for k in pick:
        r = meta.loc[k]
        real = cv2.cvtColor(cv2.imread(str(train_image_path(root, r.real_id, 0))), cv2.COLOR_BGR2RGB)
        fake = cv2.cvtColor(cv2.imread(str(train_image_path(root, r.fake_id, 1))), cv2.COLOR_BGR2RGB)
        m = masks[k] if masks[k] is not None else np.zeros((SIZE, SIZE), np.float32)
        ov = fake.astype(np.float32)
        ov = ov * (1 - 0.55 * m[..., None]) + np.array([255, 0, 0], np.float32) * 0.55 * m[..., None]
        ims = [real, fake, ov.clip(0, 255).astype(np.uint8)]
        cell = Image.new("RGB", (3 * tile + 4, tile + bar), "white")
        for j, im in enumerate(ims):
            cell.paste(Image.fromarray(im).resize((tile, tile)), (j * (tile + 2), bar))
        good = r.accepted == 1
        d = ImageDraw.Draw(cell)
        d.rectangle([0, 0, cell.width, bar], fill=(40, 140, 60) if good else (190, 40, 40))
        area = "-" if np.isnan(r.area_frac) else f"{r.area_frac:.0%}"
        od = "-" if np.isnan(r.outside_mean_d) else f"{r.outside_mean_d:.1f}"
        cap = f"{r.fake_id[:6]} a={area} out={od} r={r.response:.2f}" + ("" if good else f" {r.reject_reason}")
        d.text((3, 3), cap, fill="white", font=_font(12))
        cells.append(cell)
    title = (f"pair masks: {len(acc)} accepted (32 random, green) + rejected (16 random, red)  "
             f"| real | fake | mask | area {MIN_AREA:.0%}-{MAX_AREA:.0%}, out_d<={MAX_OUTSIDE_D}")
    grid_from_cells(cells, AUDIT_JPG, title, cols=4, gap=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--align", choices=["ecc", "phase"], default=ALIGN,
                    help="ecc (default, affine refine) or phase (PHASE4 §1 translation-only, as written)")
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    ensure_dirs()
    cv2.setNumThreads(1)
    print(f"thresholds: blur σ={BLUR_SIGMA}, max_shift={MAX_SHIFT}px, min_response={MIN_RESPONSE}, "
          f"diff σ={DIFF_BLUR_SIGMA}, open {OPEN_K}, close {CLOSE_K}, feather {FEATHER_K}, "
          f"area {MIN_AREA}-{MAX_AREA}, max_outside_d={MAX_OUTSIDE_D}")
    print(f"alignment: {args.align}" + ("" if args.align == "phase" else
          f" (min_ecc_cc={MIN_ECC_CC}, min_quality_cc={MIN_QUALITY_CC}, scale {SCALE_RANGE}, max_aniso={MAX_ANISO}, max_shift={MAX_SHIFT_ECC}px, "
          f"min_coverage={MIN_COVERAGE}, flat blocks k={FLAT_K} std<{FLAT_STD})"))
    meta, masks = build_pair_masks(pd.read_csv(DATA_DIR / "pairs.csv"), root, args.align)

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for old in MASK_DIR.glob("*.png"):
        old.unlink()
    acc = meta[meta.accepted == 1]
    for k, r in acc.iterrows():
        cv2.imwrite(str(MASK_DIR / f"{r.fake_id}.png"), np.round(masks[k] * 255).astype(np.uint8))
    prior = np.mean([masks[k] for k in acc.index], axis=0).astype(np.float32) if len(acc) else \
        np.zeros((SIZE, SIZE), np.float32)
    np.save(PRIOR_NPY, prior)
    cols = ["fake_id", "real_id", "fold_fake", "fold_real", "group_id", "cos_sim", "shift_x", "shift_y",
            "response", "ecc_cc", "scale", "coverage", "area_frac", "outside_mean_d", "inside_mean_d", "accepted", "reject_reason"]
    meta[cols].to_csv(META_CSV, index=False)
    audit_grid(meta, masks, root)

    print(f"\nreal-fake pairs evaluated: {len(meta)}  unique fakes: {meta.fake_id.nunique()}  accepted: {len(acc)}")
    print(meta.reject_reason.replace("", "accepted").value_counts().to_string())
    ev = meta.outside_mean_d.dropna()
    print(f"\noutside_mean_d (aligned pairs): " + "  ".join(
        f"p{q}={np.percentile(ev, q):.2f}" for q in (10, 25, 50, 75, 90, 95)))
    if len(acc):
        print(f"accepted area_frac: mean {acc.area_frac.mean():.3f}  median {acc.area_frac.median():.3f}  "
              f"min {acc.area_frac.min():.3f}  max {acc.area_frac.max():.3f}")
        yy, xx = np.mgrid[:SIZE, :SIZE]
        w = prior / prior.sum()
        cx, cy = (w * xx).sum(), (w * yy).sum()
        sx, sy = np.sqrt((w * (xx - cx) ** 2).sum()), np.sqrt((w * (yy - cy) ** 2).sum())
        print(f"mask prior: centre (x={cx:.1f}, y={cy:.1f}) px  std (x={sx:.1f}, y={sy:.1f}) px  peak {prior.max():.2f}")
    print(f"\nwrote {MASK_DIR}/ ({len(acc)} PNG), {META_CSV}, {PRIOR_NPY}, {AUDIT_JPG}")


if __name__ == "__main__":
    main()
