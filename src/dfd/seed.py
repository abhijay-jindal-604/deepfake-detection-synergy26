"""Seeding helpers: one global seed (42) and deterministic DataLoader workers."""
from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def seed_everything(seed: int = SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=False + deterministic algorithms -> repeatable runs (a bit slower)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own reproducible numpy/random stream."""
    import torch

    s = torch.initial_seed() % 2**32  # differs per worker, fixed by the loader's generator
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int = SEED):
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
