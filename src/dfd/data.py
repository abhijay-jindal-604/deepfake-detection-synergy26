"""Datasets and loaders. Images are returned as uint8 CHW tensors; normalisation happens inside the model."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .corruptions import SIZE, corrupt, setting_seed
from .paths import DATA_DIR, train_image_path
from .seed import make_generator


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def load_folds() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "folds.csv", dtype={"image_id": str})


class FaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, aug=None):
        self.ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.paths = [train_image_path(root, i, l) for i, l in zip(self.ids, self.labels)]
        self.aug = aug

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        img = read_rgb(self.paths[i])
        if self.aug is not None:
            img = self.aug(img)
        return to_tensor(img), torch.tensor(self.labels[i], dtype=torch.float32)


class ValCDataset(Dataset):
    """Flat index over (setting, image): one loader covers the whole val-C suite."""

    def __init__(self, df: pd.DataFrame, root: Path, settings: list[tuple[str, int]]):
        self.base = FaceDataset(df, root)
        self.settings = settings
        self.n = len(self.base)

    def __len__(self) -> int:
        return self.n * len(self.settings)

    def __getitem__(self, k):
        s_idx, i = divmod(k, self.n)
        name, sev = self.settings[s_idx]
        img = read_rgb(self.base.paths[i])
        img = corrupt(img, name, sev, setting_seed(self.base.ids[i], name, sev))
        return to_tensor(img), torch.tensor(self.base.labels[i], dtype=torch.float32)


def _worker_init(worker_id: int) -> None:
    import random

    info = torch.utils.data.get_worker_info()
    s = torch.initial_seed() % 2**31  # unique per worker, fixed by the loader's generator
    np.random.seed(s)
    random.seed(s)
    ds = info.dataset
    aug = getattr(ds, "aug", None)
    if aug is not None and hasattr(aug, "reseed"):
        aug.reseed(s)


def make_loader(ds: Dataset, batch_size: int, train: bool, workers: int, seed: int = 42) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0 and train,
        worker_init_fn=_worker_init if workers > 0 else None,
        generator=make_generator(seed),
    )
