"""Self-Blended Image (SBI) pseudo-fakes from REAL training images (docs/PHASE4.md section 3.1).

A real image is blended with a slightly altered copy of itself (colour, frequency and small geometric
changes) through a soft, elastically deformed ellipse placed at a face-position prior. The result
carries generic face-swap traces (blend seam, colour/sharpness mismatch, slight misalignment) with no
generator involved. Label = fake.

Rules: input must be a REAL training image. No face detector. No held-out corruption types
(no motion/zoom/glass blur, shot/speckle/ISO noise, fog, sketch/emboss, posterise): the only ops here
are colour shifts, sharpen, isotropic Gaussian blur, down/up-scaling and an affine warp.
TrainAug is NOT applied here; the dataset plumbing applies it afterwards, jointly with the mask.

All randomness comes from the `rng` argument (np.random.Generator); no global RNG is touched.

Audit: python -m dfd.sbi --audit  -> outputs/audit/sbi_examples.jpg (needs $DF_DATA_ROOT).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import AUDIT_DIR, DATA_DIR

SIZE = 224
RATIOS = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)
FREQ_OPS = ("sharpen", "blur", "rescale")

# Default prior (used when data/mask_prior.npy is absent): image centre, axes 0.22*W x 0.28*H.
DEFAULT_CENTRE = (0.5, 0.5)
DEFAULT_AXES = (0.22, 0.28)
# Sane ranges for a prior derived from data (fractions of W / H).
AX_MIN, AX_MAX = 0.12, 0.40
CENTRE_MIN, CENTRE_MAX = 0.25, 0.75
# Near-greyscale guard: mean over pixels of (max - min over RGB channels), uint8 units. Calibrated on
# the 3,478 train reals (4 Nov): < 8 is visually greyscale or faint sepia (201 images, 5.8%; 78 are
# exactly grey, < 1); from ~8 upwards faint real colour appears. For these sources the target gets no
# hue/sat or RGB shift (a coloured patch in a grey photo would be a trivial, unrealistic SBI cue).
GREY_SPREAD_THR = 8.0


@dataclass(frozen=True)
class FacePrior:
    """Face-region prior for a 224x224 image, in pixels. Plain floats -> picklable."""

    cx: float
    cy: float
    ax: float
    ay: float
    source: str = "default"

    @classmethod
    def default(cls, size: int = SIZE) -> "FacePrior":
        return cls(DEFAULT_CENTRE[0] * size, DEFAULT_CENTRE[1] * size,
                   DEFAULT_AXES[0] * size, DEFAULT_AXES[1] * size, "default")

    @classmethod
    def load(cls, path: str | Path | None = None, size: int = SIZE) -> "FacePrior":
        """From a 2-D non-negative heat map (data/mask_prior.npy): centre = intensity-weighted
        centroid, axes = 2 x weighted std (= semi-axes of a uniform ellipse); centre clipped to
        [0.25, 0.75], axes to [0.12, 0.40] of the image. Missing/unusable file, or a map whose
        spread exceeds 0.40 (not face-like), -> default (centre, 0.22*W x 0.28*H)."""
        p = Path(path) if path is not None else DATA_DIR / "mask_prior.npy"
        if not p.is_file():
            return cls.default(size)
        try:
            m = np.load(p, allow_pickle=False).astype(np.float64)
        except Exception:
            return cls.default(size)
        m = np.squeeze(m)
        if m.ndim != 2 or not np.isfinite(m).all():
            return cls.default(size)
        m = np.clip(m, 0, None)
        tot = m.sum()
        if tot <= 0:
            return cls.default(size)
        h, w = m.shape
        ys, xs = np.mgrid[0:h, 0:w]
        xs = (xs + 0.5) / w  # fractions of the image
        ys = (ys + 0.5) / h
        cx = (m * xs).sum() / tot
        cy = (m * ys).sum() / tot
        sx = np.sqrt((m * (xs - cx) ** 2).sum() / tot)
        sy = np.sqrt((m * (ys - cy) ** 2).sum() / tot)
        if max(2 * sx, 2 * sy) > AX_MAX:
            # Spread wider than any face blob (a uniform map gives 0.58): the map is not a usable
            # face-position prior (e.g. built from too few pair masks) -> fall back to the default.
            d = cls.default(size)
            return cls(d.cx, d.cy, d.ax, d.ay, f"default (rejected {p.name}: semi-axes {2*sx:.2f}x{2*sy:.2f} > {AX_MAX})")
        cx, cy = np.clip([cx, cy], CENTRE_MIN, CENTRE_MAX)
        ax, ay = np.clip([2 * sx, 2 * sy], AX_MIN, AX_MAX)
        return cls(float(cx * size), float(cy * size), float(ax * size), float(ay * size), str(p.name))


# ------------------------------------------------------------------ target (altered copy)
def channel_spread(img: np.ndarray) -> float:
    """Mean per-pixel channel spread (max - min over RGB) of a uint8 HxWx3 image."""
    return float(cv2.subtract(img.max(axis=2), img.min(axis=2)).mean())


def is_near_grey(img: np.ndarray, thr: float = GREY_SPREAD_THR) -> bool:
    return channel_spread(img) < thr


def _colour(img: np.ndarray, rng: np.random.Generator, grey: bool = False) -> np.ndarray:
    """RGB shift +-20, hue +-10 deg (OpenCV H units = deg/2), saturation +-10, brightness/contrast +-0.1.
    Implemented with 256-entry lookup tables (fast, exact on uint8).
    grey=True: hue/sat and RGB shifts are skipped (their rng draws still happen, so the rest of the
    draw sequence does not depend on the guard); brightness/contrast still apply."""
    x = img
    do_hs = rng.random() < 0.5
    if do_hs:
        dh = int(round(rng.uniform(-10, 10) / 2))
        ds = int(round(rng.uniform(-10, 10)))
    if do_hs and not grey:  # hue / saturation in HSV (uint8 OpenCV: H in [0,180))
        v = np.arange(256)
        lut_h = ((v + dh) % 180).astype(np.uint8)  # H only takes values < 180
        lut_s = np.clip(v + ds, 0, 255).astype(np.uint8)
        h, s_, v_ = cv2.split(cv2.cvtColor(x, cv2.COLOR_RGB2HSV))
        x = cv2.cvtColor(cv2.merge([cv2.LUT(h, lut_h), cv2.LUT(s_, lut_s), v_]), cv2.COLOR_HSV2RGB)
    shift = rng.uniform(-20, 20, size=3) if rng.random() < 0.7 else np.zeros(3)
    if grey:
        shift = np.zeros(3)
    if rng.random() < 0.7:
        alpha = 1.0 + rng.uniform(-0.1, 0.1)          # contrast (about the image mean)
        beta = 255.0 * rng.uniform(-0.1, 0.1)          # brightness
    else:
        alpha, beta = 1.0, 0.0
    mean = float(x.mean())
    v = np.arange(256, dtype=np.float64)[:, None]
    lut = np.clip(np.rint((v - mean) * alpha + mean + beta + shift[None, :]), 0, 255).astype(np.uint8)
    return cv2.LUT(x, lut.reshape(256, 1, 3))


def _frequency(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    op = FREQ_OPS[int(rng.integers(len(FREQ_OPS)))]
    if op == "sharpen":
        a = rng.uniform(0.3, 1.0)
        k = np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], np.float32)
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT), f"sharpen{a:.2f}"
    if op == "blur":
        s = rng.uniform(0.5, 1.5)
        return cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT), f"blur{s:.2f}"
    f = rng.uniform(0.5, 0.9)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), f"rescale{f:.2f}"


def _geometry(img: np.ndarray, rng: np.random.Generator, centre: tuple[float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    s = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    M = cv2.getRotationMatrix2D(centre, 0.0, s)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ mask
_GRID: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _base_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if (h, w) not in _GRID:  # per-process cache (rebuilt lazily in each DataLoader worker)
        _GRID[(h, w)] = tuple(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)))
    return _GRID[(h, w)]


def _elastic(mask: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    """Own elastic warp: coarse random displacement grid, bicubic-upsampled (smooth), cv2.remap."""
    h, w = mask.shape
    g = int(rng.integers(4, 8))
    d = rng.uniform(-amp, amp, (g, g, 2)).astype(np.float32)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = _base_grid(h, w)
    return cv2.remap(mask, xs + d[..., 0], ys + d[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _mask(h: int, w: int, rng: np.random.Generator, prior: FacePrior) -> np.ndarray:
    sx, sy = w / SIZE, h / SIZE  # prior is in 224-px units
    cx = prior.cx * sx + rng.uniform(-0.08, 0.08) * w
    cy = prior.cy * sy + rng.uniform(-0.08, 0.08) * h
    ax = prior.ax * sx * rng.uniform(0.8, 1.2)
    ay = prior.ay * sy * rng.uniform(0.8, 1.2)
    ang = rng.uniform(-15, 15)
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(2, int(round(ax))), max(2, int(round(ay)))),
                ang, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    m = _elastic(m, rng, amp=rng.uniform(0.04, 0.10) * min(ax, ay) * 2)
    k = int(rng.integers(2, 13)) * 2 + 1  # odd kernel 5..25
    m = cv2.GaussianBlur(m, (k, k), 0, borderType=cv2.BORDER_CONSTANT)
    return np.clip(m, 0.0, 1.0)


# ------------------------------------------------------------------ main entry
def make_sbi_info(img: np.ndarray, rng: np.random.Generator, prior: FacePrior | None = None
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """make_sbi plus a dict {r, freq} describing the draw (for audits/logging)."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got {img.dtype} {img.shape}")
    prior = prior or FacePrior.default()
    h, w = img.shape[:2]
    grey = is_near_grey(img)
    target = _colour(img, rng, grey)
    target, fop = _frequency(target, rng)
    target = _geometry(target, rng, (prior.cx * w / SIZE, prior.cy * h / SIZE))
    mask = _mask(h, w, rng, prior)
    r = float(RATIOS[int(rng.integers(len(RATIOS)))])
    a = (mask * np.float32(r)).astype(np.float32)
    # out = a*target + (1-a)*source, rounded and saturated to uint8 (cv2.blendLinear does exactly this)
    out = cv2.blendLinear(target, img, a, np.float32(1.0) - a)
    return out, a, {"r": r, "freq": fop, "grey": grey}


def make_sbi(img: np.ndarray, rng: np.random.Generator, prior: FacePrior | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """img uint8 HxWx3 RGB (a REAL training image) -> (fake uint8 HxWx3, mask float32 HxW in [0,1]),
    where mask = r * blend_mask is the effective per-pixel blend weight. Deterministic given rng state."""
    out, mask, _ = make_sbi_info(img, rng, prior)
    return out, mask


# ------------------------------------------------------------------ audit
def _audit(n: int = 32, seed: int = 42, cols: int = 4) -> None:
    import pandas as pd

    from .paths import resolve_data_root, train_image_path

    cv2.setNumThreads(1)
    root = resolve_data_root(None)
    meta = pd.read_csv(root / "train_metadata.csv")
    reals = meta[meta["label"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    prior = FacePrior.load()

    def _read(i):
        q = train_image_path(root, reals.loc[i, "image_id"], 0)
        im = cv2.cvtColor(cv2.imread(str(q), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return im if im.shape[:2] == (SIZE, SIZE) else cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    # random reals, but guarantee >= n_grey near-greyscale sources so the colour guard is visible
    n_grey = min(4, n)
    order = rng.permutation(len(reals))
    grey_idx, other_idx = [], []
    for i in order:
        g = is_near_grey(_read(i))
        if g and len(grey_idx) < n_grey:
            grey_idx.append(i)
        elif not g and len(other_idx) < n - n_grey:
            other_idx.append(i)
        if len(grey_idx) == n_grey and len(other_idx) == n - n_grey:
            break
    idx = rng.permutation(np.array(grey_idx + other_idx))
    print(f"prior: {prior}")

    imgs = [_read(i) for i in idx]

    gen_rng = np.random.default_rng(seed)
    results = [make_sbi_info(im, gen_rng, prior) for im in imgs]

    # timing: single core, many draws
    t_rng = np.random.default_rng(seed + 1)
    reps = 20
    for im in imgs[:4]:
        make_sbi(im, t_rng, prior)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        for im in imgs:
            make_sbi(im, t_rng, prior)
    ms = (time.perf_counter() - t0) * 1000 / (reps * len(imgs))
    print(f"make_sbi: {ms:.2f} ms/img (1 thread, {reps * len(imgs)} calls, {SIZE}x{SIZE})")

    cap = 16
    tiles = []
    for im, (fake, mask, info) in zip(imgs, results):
        m = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        row = np.concatenate([im, fake, m], axis=1)
        bar = np.full((cap, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"src | SBI r={info['r']:.2f} {info['freq']}{' GREY' if info['grey'] else ''} | mask", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        tile = np.concatenate([bar, row], axis=0)
        tiles.append(np.pad(tile, ((3, 3), (3, 3), (0, 0)), constant_values=128))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    grid = np.concatenate(rows, axis=0)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "sbi_examples.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {out} ({grid.shape[1]}x{grid.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", action="store_true", help="write outputs/audit/sbi_examples.jpg and time make_sbi")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.audit:
        _audit(args.n, args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
