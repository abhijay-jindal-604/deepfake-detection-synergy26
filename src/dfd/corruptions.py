"""Deterministic corruption suite ("val-C").

corrupt(img, name, severity, seed) -> img
  img: HxWx3 uint8 RGB (any size; output is resized to 224x224)
  severity: 1, 2 or 3
  seed: int; same inputs -> bit-identical output

SEEN corruptions may also be used as training augmentation (augment.py).
HELD_OUT corruptions must never be used in training: they measure generalisation to unseen degradations.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

SIZE = 224


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ compression / resolution
def jpeg(img, s, rng):
    q = {1: 60, 2: 35, 3: 15}[s]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def downscale(img, s, rng):
    f = {1: 0.6, 2: 0.4, 3: 0.25}[s]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def pixelate(img, s, rng):
    b = {1: 4, 2: 8, 3: 12}[s]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ blur
def gaussian_blur(img, s, rng):
    sigma = {1: 1.0, 2: 2.0, 3: 3.0}[s]
    return cv2.GaussianBlur(img, (0, 0), sigma)


def motion_blur(img, s, rng):  # held out
    k = {1: 7, 2: 11, 3: 15}[s]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ noise
def gaussian_noise(img, s, rng):
    sigma = {1: 8, 2: 16, 3: 28}[s]
    return _u8(img.astype(np.float32) + rng.normal(0, sigma, img.shape))


def shot_noise(img, s, rng):  # held out; Poisson photon noise
    lam = {1: 60, 2: 25, 3: 10}[s]
    x = img.astype(np.float32) / 255.0
    return _u8(rng.poisson(x * lam) / lam * 255.0)


def speckle_noise(img, s, rng):  # held out; multiplicative noise
    c = {1: 0.15, 2: 0.3, 3: 0.5}[s]
    x = img.astype(np.float32)
    return _u8(x + x * rng.normal(0, c, img.shape))


# ------------------------------------------------------------------ weather / overlay
def snow(img, s, rng):
    density = {1: 0.002, 2: 0.005, 3: 0.010}[s]
    h, w = img.shape[:2]
    layer = (rng.random((h, w)) < density).astype(np.float32)
    layer = cv2.dilate(layer, np.ones((2, 2), np.uint8))
    k = 7
    kernel = np.zeros((k, k), np.float32)
    kernel[:, k // 2] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(-30, 30), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    layer = cv2.filter2D(layer, -1, kernel / kernel.sum() * 3.0)
    layer = np.clip(layer, 0, 1)[..., None]
    x = img.astype(np.float32)
    x = x * (1 - 0.15 * s / 3) + 255 * 0.15 * s / 3 * 0.5  # slight washed-out look
    return _u8(x * (1 - layer) + 250 * layer)


def fog(img, s, rng):  # held out; low-frequency haze
    amount = {1: 0.3, 2: 0.5, 3: 0.7}[s]
    h, w = img.shape[:2]
    haze = np.zeros((h, w), np.float32)
    for octave, wgt in ((4, 0.5), (8, 0.3), (16, 0.2)):
        haze += wgt * cv2.resize(rng.random((octave, octave)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    haze = (haze - haze.min()) / (np.ptp(haze) + 1e-6)
    a = (amount * (0.6 + 0.4 * haze))[..., None]
    return _u8(img.astype(np.float32) * (1 - a) + 215 * a)


# ------------------------------------------------------------------ occlusion
def cutout(img, s, rng):
    n = {1: 1, 2: 3, 3: 6}[s]
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        sz = int(rng.integers(h // 8, h // 4))
        y, x = int(rng.integers(0, h - sz)), int(rng.integers(0, w - sz))
        out[y : y + sz, x : x + sz] = int(rng.integers(40, 220))
    return out


def block_patches(img, s, rng):
    n = {1: 5, 2: 15, 3: 30}[s]
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(n):
        sz = int(rng.integers(h // 28, h // 14))
        y, x = int(rng.integers(0, h - sz)), int(rng.integers(0, w - sz))
        out[y : y + sz, x : x + sz] = rng.integers(0, 256, 3)
    return out


# ------------------------------------------------------------------ colour
def greyscale(img, s, rng):
    a = {1: 0.5, 2: 0.8, 3: 1.0}[s]
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return _u8(img * (1 - a) + g * a)


def brightness_up(img, s, rng):
    d = {1: 0.2, 2: 0.35, 3: 0.5}[s]
    return _u8(img.astype(np.float32) + 255 * d)


def brightness_down(img, s, rng):
    d = {1: 0.2, 2: 0.35, 3: 0.5}[s]
    return _u8(img.astype(np.float32) - 255 * d)


def contrast(img, s, rng):
    f = {1: 0.7, 2: 0.5, 3: 0.3}[s]
    x = img.astype(np.float32)
    return _u8((x - x.mean()) * f + x.mean())


def colour_cast(img, s, rng):
    m = {1: 20, 2: 40, 3: 60}[s]
    shift = rng.uniform(-1, 1, 3)
    shift = shift / (np.abs(shift).max() + 1e-6) * m
    return _u8(img.astype(np.float32) + shift)


# ------------------------------------------------------------------ filters (held out)
def sketch(img, s, rng):
    a = {1: 0.5, 2: 0.75, 3: 1.0}[s]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_blur = cv2.GaussianBlur(255 - g, (0, 0), 3)
    dodge = cv2.divide(g, 255 - inv_blur, scale=256)  # colour-dodge "pencil sketch"
    sk = cv2.cvtColor(dodge, cv2.COLOR_GRAY2RGB).astype(np.float32)
    return _u8(img * (1 - a) + sk * a)


def posterise(img, s, rng):
    bits = {1: 4, 2: 3, 3: 2}[s]
    shift = 8 - bits
    return ((img >> shift) << shift).astype(np.uint8)


SEEN = {
    "jpeg": jpeg, "downscale": downscale, "pixelate": pixelate, "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise, "snow": snow, "cutout": cutout, "block_patches": block_patches,
    "greyscale": greyscale, "brightness_up": brightness_up, "brightness_down": brightness_down,
    "contrast": contrast, "colour_cast": colour_cast,
}
HELD_OUT = {
    "motion_blur": motion_blur, "shot_noise": shot_noise, "speckle_noise": speckle_noise,
    "fog": fog, "sketch": sketch, "posterise": posterise,
}
ALL = {**SEEN, **HELD_OUT}
SEVERITIES = (1, 2, 3)


def setting_seed(image_id: str, name: str, severity: int) -> int:
    return zlib.crc32(f"{image_id}|{name}|{severity}".encode())


def corrupt(img: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    out = ALL[name](img, severity, rng)
    assert out.shape == (SIZE, SIZE, 3) and out.dtype == np.uint8, (name, out.shape, out.dtype)
    return out


def all_settings() -> list[tuple[str, int]]:
    return [(n, s) for n in ALL for s in SEVERITIES]
