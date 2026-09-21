"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

ASSIGNMENT_SEED: int = 6304


def set_seed(seed: int = ASSIGNMENT_SEED, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a derived seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = ASSIGNMENT_SEED) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
