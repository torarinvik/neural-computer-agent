"""Open-ended world generator with a built-in sealed split (F245).

Samples game-family configurations from the verifier's supported
component space. The F230 sealed trio (blink, oneway, lever) is
excluded entirely -- it remains a separate, older seal. Constraints
mirror FamilyConfig.validate(): deceptive needs avoid; every world
carries at least one reward-bearing component.

The generator's honesty device is the SPLIT: `generate(seed)` returns
(dev, sealed) halves, and the sealed half's manifest (config dicts +
SHA256 of their canonical string) is committed to the repository
BEFORE any probing. Sealed configs must never be instantiated by any
probe until an explicit unseal decision is recorded.
"""

from __future__ import annotations

import hashlib
import json

import torch

from experiments.games_amodal.game_family import FamilyConfig

SPACE = {
    "collect": (0, 1, 2, 3),
    "avoid": (0, 1, 2, 3),
    "pursue": (0, 1),
    "intercept": (0, 1, 2),
    "delayed": (0, 2, 3, 4, 5),
    "resource": (0, 1, 2),
    "deceptive": (0, 1, 2),
}


def sample_config(gen: torch.Generator) -> dict:
    def pick(name):
        options = SPACE[name]
        return options[int(torch.randint(0, len(options), (1,),
                                         generator=gen))]

    while True:
        c = {name: pick(name) for name in SPACE}
        if c["deceptive"] and not c["avoid"]:
            continue
        rewarding = (c["collect"] or c["intercept"] or c["delayed"])
        if not rewarding:
            continue
        active = sum(1 for v in c.values() if v)
        if active < 1 or active > 4:
            continue
        try:
            FamilyConfig(**c).validate()
        except ValueError:
            continue
        return c


def name_of(c: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(c.items()) if v)


def generate(seed: int, n: int = 40):
    gen = torch.Generator().manual_seed(seed)
    seen, configs = set(), []
    while len(configs) < n:
        c = sample_config(gen)
        key = name_of(c)
        if key not in seen:
            seen.add(key)
            configs.append(c)
    dev, sealed = configs[: n // 2], configs[n // 2:]
    manifest = json.dumps(sealed, sort_keys=True)
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    return dev, sealed, digest
