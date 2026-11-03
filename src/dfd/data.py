"""Datasets and loaders. Images are returned as uint8 CHW tensors; normalisation happens inside the model.

Phase 4 additions (docs/PHASE4.md):
- Mask targets (§2): with `with_mask=True` a training item is (img, label, mask, mask_valid); mask is float32
  HxW at the data resolution (224), mask_valid a float32 0/1 scalar.
    real                                   -> zeros, valid 1 ("nothing manipulated here")
    fake with an accepted pair mask        -> data/masks/<fake_id>.png / 255, valid 1   (data.pair_masks)
    any other fake                         -> zeros, valid 0 (ignored by the mask loss)
    SBI pseudo-fake                        -> its exact blend mask, valid 1
  Geometric augmentation is applied jointly to image and mask (TrainAug(img, mask)).
- SBIDataset (§3.3): fresh self-blended fakes from TRAIN-fold reals, different every epoch.
- GroupBatchSampler (§4): members of one source-photo group land in the same batch.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from .corruptions import SIZE, corrupt, setting_seed
from .paths import DATA_DIR, train_image_path
from .seed import make_generator

MASK_DIR = DATA_DIR / "masks"
MASKS_META = DATA_DIR / "masks_meta.csv"
GROUPS_CSV = DATA_DIR / "groups.csv"
_SBI_TAG = 0x5B1  # domain separator for SBI seed sequences


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img


def read_mask(path: str | Path) -> np.ndarray:
    """8-bit PNG -> float32 HxW in [0, 1] at SIZE x SIZE."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != (SIZE, SIZE):
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return m.astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def load_folds() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "folds.csv", dtype={"image_id": str})


def load_pair_masks(meta_csv: Path = MASKS_META, mask_dir: Path = MASK_DIR) -> dict[str, Path]:
    """{fake_id: mask path} for every ACCEPTED pair mask (masks_meta.csv accepted == 1)."""
    if not Path(meta_csv).is_file():
        raise FileNotFoundError(f"{meta_csv} missing: run python -m dfd.masks first (data.pair_masks: true)")
    meta = pd.read_csv(meta_csv, dtype={"fake_id": str})
    out = {}
    for fid in meta.loc[meta.accepted == 1, "fake_id"]:
        p = Path(mask_dir) / f"{fid}.png"
        if not p.is_file():
            raise FileNotFoundError(f"accepted mask {p} missing")
        out[fid] = p
    return out


class FaceDataset(Dataset):
    """with_mask=False: (img, label) exactly as before Phase 4. with_mask=True: (img, label, mask, valid)."""

    def __init__(self, df: pd.DataFrame, root: Path, aug=None, masks: dict[str, Path] | None = None,
                 with_mask: bool = False):
        self.ids = df.image_id.tolist()
        self.labels = df.label.astype(int).tolist()
        self.paths = [train_image_path(root, i, l) for i, l in zip(self.ids, self.labels)]
        self.aug = aug
        self.with_mask = bool(with_mask)
        masks = masks or {}
        # only fakes can carry a pair mask (masks are in the fake's frame)
        self.mask_paths = [masks.get(i) if l == 1 else None for i, l in zip(self.ids, self.labels)]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        img = read_rgb(self.paths[i])
        if not self.with_mask:
            if self.aug is not None:
                img = self.aug(img)
            return to_tensor(img), torch.tensor(self.labels[i], dtype=torch.float32)
        mp = self.mask_paths[i]
        if mp is not None:
            mask, valid = read_mask(mp), 1.0
        else:  # real -> all-zero valid target; fake without a pair mask -> ignored
            mask, valid = np.zeros((SIZE, SIZE), np.float32), (1.0 if self.labels[i] == 0 else 0.0)
        if self.aug is not None:
            img, mask = self.aug(img, mask)
        return (to_tensor(img), torch.tensor(self.labels[i], dtype=torch.float32),
                torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)), torch.tensor(valid, dtype=torch.float32))


class SBIDataset(Dataset):
    """Virtual dataset of round(ratio * n_real) self-blended pseudo-fakes (PHASE4 §3.3).

    Item i of epoch e: real image perm_e[i mod n_real] (perm_e = permutation drawn from (seed, e)), blended by
    make_sbi with an rng seeded from (seed, e, i). Every index is fetched exactly once per epoch, so the
    worker id is not needed for uniqueness and the SBI fakes do not depend on the number of workers;
    TrainAug (applied jointly to image + mask afterwards) draws from the worker's own stream as usual.
    `df_real` must hold TRAIN-fold real images only (enforced: label 0).

    set_epoch(e) writes into a shared-memory tensor, so persistent DataLoader workers see it; call it
    before iterating the loader for epoch e (samplers only hand out epoch-e indices after that).
    """

    def __init__(self, df_real: pd.DataFrame, root: Path, ratio: float, aug=None, seed: int = 42, prior=None):
        from .sbi import FacePrior

        if not (df_real.label.astype(int) == 0).all():
            raise ValueError("SBIDataset takes real images only")
        if ratio <= 0:
            raise ValueError(f"sbi.ratio must be > 0, got {ratio}")
        self.ids = df_real.image_id.tolist()
        self.paths = [train_image_path(root, i, 0) for i in self.ids]
        self.n_real = len(self.ids)
        self.n = int(round(ratio * self.n_real))
        self.aug = aug
        self.seed = int(seed)
        self.prior = prior if prior is not None else FacePrior.load()
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._perm_cache: tuple[int, np.ndarray] | None = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return self.n

    def _perm(self, ep: int) -> np.ndarray:
        if self._perm_cache is None or self._perm_cache[0] != ep:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, ep, _SBI_TAG]))
            self._perm_cache = (ep, rng.permutation(self.n_real))
        return self._perm_cache[1]

    def source_index(self, i: int, ep: int | None = None) -> int:
        ep = int(self._epoch[0]) if ep is None else ep
        return int(self._perm(ep)[i % self.n_real])

    def __getitem__(self, i):
        from .sbi import make_sbi

        ep = int(self._epoch[0])
        img = read_rgb(self.paths[self.source_index(i, ep)])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, ep, int(i), _SBI_TAG]))
        fake, mask = make_sbi(img, rng, self.prior)
        if self.aug is not None:
            fake, mask = self.aug(fake, mask)
        return (to_tensor(fake), torch.tensor(1.0), torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)),
                torch.tensor(1.0))


class TrainSet(ConcatDataset):
    """FaceDataset + SBIDataset sharing ONE TrainAug, exposed as `.aug` so _worker_init reseeds it."""

    def __init__(self, datasets, aug=None):
        super().__init__(datasets)
        self.aug = aug

    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)


class GroupBatchSampler(Sampler):
    """Pair-aware batches (PHASE4 §4), deterministic per epoch.

    Units: each source-photo group (data/groups.csv group_id) with >= 2 members in `df` is shuffled and split
    into chunks of <= max_per_group; every other image, and every extra index (n_extra, e.g. SBI items at
    positions len(df)...len(df)+n_extra-1 of a TrainSet), is a singleton unit. Units are shuffled and packed
    greedily: a unit that does not fit the current batch is deferred to the start of the next one, so a
    chunk is never split, and singletons fill the remaining room. Leftovers at the end are chunked and the
    last incomplete batch is dropped: len = N // batch_size, like DataLoader(drop_last=True).
    """

    def __init__(self, df: pd.DataFrame, batch_size: int, max_per_group: int = 4, seed: int = 42, n_extra: int = 0,
                 groups_csv: Path = GROUPS_CSV):
        g = pd.read_csv(groups_csv, dtype={"image_id": str}).set_index("image_id").group_id
        gid = g.reindex(df.image_id.astype(str)).to_numpy()
        if pd.isna(gid).any():
            raise ValueError("images missing from groups.csv")
        self.batch_size = int(batch_size)
        self.max_per_group = int(max_per_group)
        self.seed = int(seed)
        self.n = len(df) + int(n_extra)
        by_group: dict[int, list[int]] = {}
        for i, k in enumerate(gid.astype(int)):
            by_group.setdefault(int(k), []).append(i)
        self.groups = [m for m in by_group.values() if len(m) >= 2]
        in_groups = {i for m in self.groups for i in m}
        self.singles = [i for i in range(len(df)) if i not in in_groups] + list(range(len(df), self.n))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n // self.batch_size

    def batches(self, epoch: int) -> list[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), 0x6B5]))
        units: list[list[int]] = []
        for m in self.groups:
            m = [m[j] for j in rng.permutation(len(m))]
            units += [m[k:k + self.max_per_group] for k in range(0, len(m), self.max_per_group)]
        units += [[i] for i in self.singles]
        order = rng.permutation(len(units))
        bs, out, cur, deferred = self.batch_size, [], [], []
        for u in (units[j] for j in order):
            if len(cur) + len(u) <= bs:
                cur += u
            else:
                deferred.append(u)
            while len(cur) == bs:
                out.append(cur)
                cur, rest = [], []
                for d in deferred:
                    if len(cur) + len(d) <= bs:
                        cur += d
                    else:
                        rest.append(d)
                deferred = rest
        tail = cur + [i for d in deferred for i in d]
        out += [tail[k:k + bs] for k in range(0, len(tail) - bs + 1, bs)]
        assert len(out) == len(self)
        return out

    def __iter__(self):
        b = self.batches(self.epoch)
        self.epoch += 1  # if set_epoch is never called, consecutive epochs still differ
        return iter(b)


def train_has_masks(cfg: dict) -> bool:
    """Training items carry (mask, valid) when the mask head or SBI is on."""
    return bool((cfg.get("model") or {}).get("mask_head")) or bool(cfg.get("sbi"))


def build_train_set(cfg: dict, df_tr: pd.DataFrame, root: Path, aug, seed: int):
    """(dataset, df_fit, sbi_dataset | None). Without the Phase 4 keys: FaceDataset(df_tr, root, aug), as before.

    data.drop_real_fakes (SBI-only diagnostic): real fakes are removed; only train reals + SBI remain.
    SBI sources are always the TRAIN fold's reals."""
    d = cfg.get("data") or {}
    s = cfg.get("sbi") or None
    df_fit = df_tr
    if d.get("drop_real_fakes"):
        if not s:
            raise ValueError("data.drop_real_fakes needs sbi (else there are no fakes to train on)")
        df_fit = df_tr[df_tr.label.astype(int) == 0].reset_index(drop=True)
    if not train_has_masks(cfg):
        if d.get("pair_masks"):
            raise ValueError("data.pair_masks needs model.mask_head: true")
        return FaceDataset(df_fit, root, aug), df_fit, None
    masks = load_pair_masks() if d.get("pair_masks") else None
    ds = FaceDataset(df_fit, root, aug, masks=masks, with_mask=True)
    if not s:
        return ds, df_fit, None
    df_real = df_tr[df_tr.label.astype(int) == 0].reset_index(drop=True)
    sbi = SBIDataset(df_real, root, float(s.get("ratio", 0.3)), aug, seed)
    return TrainSet([ds, sbi], aug), df_fit, sbi


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


def make_loader(ds: Dataset, batch_size: int, train: bool, workers: int, seed: int = 42,
                batch_sampler: Sampler | None = None) -> DataLoader:
    if batch_sampler is not None:  # GroupBatchSampler: it owns batching, shuffling and drop_last
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0 and train,
            worker_init_fn=_worker_init if workers > 0 else None,
            generator=make_generator(seed),
        )
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
