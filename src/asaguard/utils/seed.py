"""Basic seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False, warn_only=True)


def seed_everything(seed: int) -> None:
    set_global_seed(seed)
