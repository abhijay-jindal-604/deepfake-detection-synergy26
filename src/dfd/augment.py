"""Training augmentation pipelines. Label-independent by design (corruptions appear in both classes).

Order: geo_colour -> ExtraAug (aug.extra_p > 0) -> SeenCorruptions (aug.seen_corruptions_p > 0).

Held-out val-C corruptions (motion blur, shot/speckle noise, fog, sketch, posterise) must never appear in
training, not even as "similar" library transforms: see HELD_OUT_DENYLIST and docs/PHASE2.md section 2.3.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

from .corruptions import (
    SEEN,
    SEVERITIES,
    block_patches_p,
    colour_cast_p,
    corrupt,
    corrupt_continuous,
    downscale_p,
    gaussian_blur_p,
    greyscale_p,
    jpeg_p,
    pixelate_p,
    snow_p,
)
from .seed import SEED

# albumentations class names that are (or behave like) a held-out val-C corruption
HELD_OUT_DENYLIST = {
    "MotionBlur", "ZoomBlur", "ISONoise", "ShotNoise", "MultiplicativeNoise",
    "RandomFog", "Posterize", "Emboss", "Superpixels",
    # look-alikes added 3 Nov (review): directional/glass blur, low-frequency veils
    "AdvancedBlur", "GlassBlur", "PlasmaBrightnessContrast", "PlasmaShadow",
}


def _derive(seed: int, k: int) -> int:
    """Independent child seed k of `seed` (for components added after Phase 1)."""
    return int(np.random.SeedSequence([int(seed) % 2**32, k]).generate_state(1)[0] % 2**31)


def _walk(t):
    yield t
    for child in getattr(t, "transforms", None) or []:
        yield from _walk(child)


def assert_no_heldout(pipeline) -> None:
    """Raise if any albumentations transform in `pipeline` (recursing into Compose/OneOf/SomeOf) is denied."""
    for t in _walk(pipeline):
        name = type(t).__name__
        if name in HELD_OUT_DENYLIST:
            raise AssertionError(f"held-out transform {name} found in training pipeline")


# ------------------------------------------------------------------ seen-corruption augmentation
class SeenCorruptions:
    """With prob p, apply 1..n_max random SEEN val-C corruptions (never the held-out ones).

    weights: {name: w} (unspecified names weight 1); sev_mode "discrete" draws severity from {1,2,3}
    (the val-C functions), "continuous" draws s ~ U(sev_range) and uses corrupt_continuous.
    With defaults the RNG draw sequence equals the Phase 1 implementation exactly.
    """

    def __init__(
        self,
        p: float = 0.5,
        names: list[str] | None = None,
        weights: dict[str, float] | None = None,
        n_max: int = 2,
        sev_mode: str = "discrete",
        sev_range: tuple[float, float] = (0.5, 3.5),
        seed: int = SEED,
    ):
        self.p = p
        self.names = list(names) if names else list(SEEN)
        bad = [n for n in list(self.names) + list(weights or {}) if n not in SEEN]
        if bad:
            raise ValueError(f"unknown / non-SEEN corruption names: {bad}; allowed: {list(SEEN)}")
        w = np.array([float((weights or {}).get(n, 1.0)) for n in self.names])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"bad corr_weights: {weights}")
        self.probs = None if np.all(w == w[0]) else w / w.sum()  # None -> uniform, Phase 1 draw path
        if not 1 <= int(n_max):
            raise ValueError("n_max must be >= 1")
        self.n_max = int(n_max)
        if sev_mode not in ("discrete", "continuous"):
            raise ValueError(f"sev_mode must be discrete|continuous, got {sev_mode!r}")
        self.sev_mode = sev_mode
        self.sev_range = (float(sev_range[0]), float(sev_range[1]))
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        for _ in range(int(self.rng.integers(1, self.n_max + 1))):
            if self.probs is None:
                name = self.names[int(self.rng.integers(len(self.names)))]
            else:
                name = self.names[int(self.rng.choice(len(self.names), p=self.probs))]
            if self.sev_mode == "discrete":
                sev = int(self.rng.choice(SEVERITIES))
                img = corrupt(img, name, sev, int(self.rng.integers(2**31)))
            else:
                s = float(self.rng.uniform(*self.sev_range))
                img = corrupt_continuous(img, name, s, int(self.rng.integers(2**31)))
        return img


# ------------------------------------------------------------------ extra seen-family augmentation (PHASE2 2.3)
_INTERP = (cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR)


def _add_noise(img, n):
    return np.clip(np.rint(img.astype(np.float32) + n), 0, 255).astype(np.uint8)


def _noise_rgb(img, rng):  # per-channel additive Gaussian, sigma up to 35 (float32 for speed)
    return _add_noise(img, rng.standard_normal(img.shape, dtype=np.float32) * np.float32(rng.uniform(3, 35)))


def _noise_lum(img, rng):  # luminance-only Gaussian (same additive noise on R, G, B)
    n = rng.standard_normal(img.shape[:2], dtype=np.float32) * np.float32(rng.uniform(3, 35))
    return _add_noise(img, n[..., None])


def _salt_pepper(img, rng):  # impulse noise (not signal-dependent)
    out = img.copy()
    m = rng.random(img.shape[:2])
    amt, salt = rng.uniform(0.005, 0.05), rng.uniform(0.3, 0.7)
    out[m < amt * salt] = 255
    out[(m >= amt * salt) & (m < amt)] = 0
    return out


def _jpeg(img, rng):
    return jpeg_p(img, rng.integers(10, 96))


def _double_jpeg(img, rng):
    return jpeg_p(jpeg_p(img, rng.integers(30, 96)), rng.integers(10, 96))


def _downscale(img, rng):
    return downscale_p(img, rng.uniform(0.2, 0.8), _INTERP[rng.integers(3)], _INTERP[rng.integers(3)])


def _pixelate(img, rng):
    return pixelate_p(img, rng.integers(2, 17))


def _gblur(img, rng):
    return gaussian_blur_p(img, rng.uniform(0.5, 3.5))


def _snow(img, rng):
    return snow_p(img, rng.uniform(0.001, 0.012), rng.uniform(0.0, 0.15), rng)


def _colour_cast(img, rng):
    return colour_cast_p(img, rng.uniform(10, 60), rng)


def _greyscale(img, rng):
    return greyscale_p(img, rng.uniform(0.5, 1.0))


def _block_patches(img, rng):
    return block_patches_p(img, rng.integers(3, 31), rng)


class ExtraAug:
    """With prob p apply ONE transform, uniformly chosen, from PHASE2 2.3's allowed column."""

    def __init__(self, p: float = 0.3, seed: int = SEED):
        self.p = p
        self.custom = [
            _noise_rgb, _noise_lum, _salt_pepper, _jpeg, _double_jpeg, _downscale, _pixelate,
            _gblur, _snow, _colour_cast, _greyscale, _block_patches,
        ]
        albu = [
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0),  # isotropic disk
            A.RandomRain(slant_range=(-10, 10), drop_length=12, drop_width=1, blur_value=3,
                         brightness_coefficient=0.85, p=1.0),
            A.RandomGamma(gamma_limit=(60, 160), p=1.0),
            A.CLAHE(clip_limit=(1.0, 4.0), p=1.0),
            A.Equalize(by_channels=False, p=1.0),
            A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(0, 0), val_shift_limit=(0, 0), p=1.0),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.2),
                            hole_width_range=(0.05, 0.2), fill="random_uniform", p=1.0),
        ]
        self.albu = [A.Compose([t], seed=_derive(seed, 100 + i)) for i, t in enumerate(albu)]
        self.n_ops = len(self.custom) + len(self.albu)
        self.rng = np.random.default_rng(seed)

    def pipelines(self) -> list:
        return list(self.albu)

    def op_names(self) -> list[str]:
        return [f.__name__.lstrip("_") for f in self.custom] + [type(c.transforms[0]).__name__ for c in self.albu]

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        for i, c in enumerate(self.albu):
            c.set_random_seed(_derive(seed, 100 + i))

    def apply_op(self, img: np.ndarray, k: int) -> np.ndarray:
        if k < len(self.custom):
            return self.custom[k](img, self.rng)
        return self.albu[k - len(self.custom)](image=img)["image"]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        return self.apply_op(img, int(self.rng.integers(self.n_ops)))


# ------------------------------------------------------------------ full training pipeline
class TrainAug:
    def __init__(self, cfg: dict):
        a = cfg.get("aug", {})
        seed = int(cfg.get("seed", SEED))
        self.geo_colour = A.Compose(
            [
                A.HorizontalFlip(p=a.get("hflip", 0.5)),
                A.Affine(
                    translate_percent=(-a.get("shift", 0.06), a.get("shift", 0.06)),
                    scale=(1 - a.get("scale", 0.1), 1 + a.get("scale", 0.1)),
                    rotate=(-a.get("rotate", 10), a.get("rotate", 10)),
                    border_mode=0,
                    mask_interpolation=cv2.INTER_LINEAR,  # soft masks (PHASE4); no effect on the image path
                    p=a.get("affine_p", 0.5),
                ),
                A.ColorJitter(*a.get("color_jitter", [0.2, 0.2, 0.2, 0.02]), p=a.get("color_jitter_p", 0.5)),
            ],
            seed=seed,
        )
        extra_p = a.get("extra_p", 0.0)
        self.extra = ExtraAug(extra_p, seed=_derive(seed, 2)) if extra_p > 0 else None
        p_corr = a.get("seen_corruptions_p", 0.0)
        self.corr = (
            SeenCorruptions(
                p_corr,
                weights=a.get("corr_weights"),
                n_max=a.get("n_max", 2),
                sev_mode=a.get("sev_mode", "discrete"),
                sev_range=tuple(a.get("sev_range", (0.5, 3.5))),
                seed=seed,
            )
            if p_corr > 0
            else None
        )
        for pl in self.pipelines():
            assert_no_heldout(pl)

    def pipelines(self) -> list:
        return [self.geo_colour] + (self.extra.pipelines() if self.extra is not None else [])

    def reseed(self, seed: int) -> None:
        """Called from each DataLoader worker so workers draw different, reproducible streams."""
        self.geo_colour.set_random_seed(seed)
        if self.corr is not None:
            self.corr.reseed(seed + 1)  # Phase 1 derivation, kept for reproducibility
        if self.extra is not None:
            self.extra.reseed(_derive(seed, 2))

    def __call__(self, img: np.ndarray, mask: np.ndarray | None = None):
        """img -> img; or (img, mask) -> (img, mask) when a float32 HxW manipulation mask is given (PHASE4).

        The mask follows the geometric ops only (flip, affine; linear interpolation, zero fill); colour
        jitter, ExtraAug and corruptions touch the image only. Passing a mask draws no extra randomness,
        so the image output is bit-identical to the mask-less call with the same RNG state."""
        if mask is None:
            img = self.geo_colour(image=img)["image"]
        else:
            r = self.geo_colour(image=img, mask=mask)
            img, mask = r["image"], r["mask"]
        if self.extra is not None:
            img = self.extra(img)
        if self.corr is not None:
            img = self.corr(img)
        return img if mask is None else (img, mask)
