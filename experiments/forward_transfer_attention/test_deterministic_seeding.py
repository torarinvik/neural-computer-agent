import random

import numpy as np
import torch

from .train import seed_everything


def test_seed_everything_repeats_all_host_rngs() -> None:
    seed_everything(263)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    seed_everything(263)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seed_everything_enables_deterministic_cudnn() -> None:
    seed_everything(263)
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
