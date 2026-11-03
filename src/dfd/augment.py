"""Training augmentation pipelines. Label-independent by design (corruptions appear in both classes)."""
from __future__ import annotations

import albumentations as A
import numpy as np

from .corruptions import SEEN, SEVERITIES, corrupt
from .seed import SEED


class SeenCorruptions:
    """With prob p, apply 1-2 random SEEN val-C corruptions (never the held-out ones)."""

    def __init__(self, p: float = 0.5, names: list[str] | None = None):
        self.p = p
        self.names = names or list(SEEN)
        self.rng = np.random.default_rng(SEED)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.p:
            return img
        for _ in range(int(self.rng.integers(1, 3))):
            name = self.names[int(self.rng.integers(len(self.names)))]
            sev = int(self.rng.choice(SEVERITIES))
            img = corrupt(img, name, sev, int(self.rng.integers(2**31)))
        return img


class TrainAug:
    def __init__(self, cfg: dict):
        a = cfg.get("aug", {})
        self.geo_colour = A.Compose(
            [
                A.HorizontalFlip(p=a.get("hflip", 0.5)),
                A.Affine(
                    translate_percent=(-a.get("shift", 0.06), a.get("shift", 0.06)),
                    scale=(1 - a.get("scale", 0.1), 1 + a.get("scale", 0.1)),
                    rotate=(-a.get("rotate", 10), a.get("rotate", 10)),
                    border_mode=0,
                    p=a.get("affine_p", 0.5),
                ),
                A.ColorJitter(*a.get("color_jitter", [0.2, 0.2, 0.2, 0.02]), p=a.get("color_jitter_p", 0.5)),
            ],
            seed=SEED,
        )
        p_corr = a.get("seen_corruptions_p", 0.0)
        self.corr = SeenCorruptions(p_corr) if p_corr > 0 else None

    def reseed(self, seed: int) -> None:
        """Called from each DataLoader worker so workers draw different, reproducible streams."""
        self.geo_colour.set_random_seed(seed)
        if self.corr is not None:
            self.corr.reseed(seed + 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self.geo_colour(image=img)["image"]
        if self.corr is not None:
            img = self.corr(img)
        return img
