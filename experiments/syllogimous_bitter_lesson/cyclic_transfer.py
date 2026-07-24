from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from experiments.syllogimous_latent_agent.data import (
    PCM_SAMPLES,
    TRAIN_PREFIXES,
    PublicEpisode,
    render_public_audio,
    render_public_card,
)
from experiments.syllogimous_realtime.environment import Action, XorShift64


def generate_cyclic_episode(seed: int, premises: int, modulus: int, *,
                            heldout: bool = False,
                            entity_count: int = 128) -> PublicEpisode:
    """Compose multi-valued cyclic transformations from public sensory cards.

    Modulus two is XOR-like parity. Larger moduli increase the information and
    composition burden per premise while retaining one deterministic binary
    query and no task-specific input to the model.
    """
    if premises < 1 or premises >= entity_count:
        raise ValueError("premises must be between 1 and entity_count - 1")
    if modulus not in {2, 4, 8}:
        raise ValueError("modulus must be one of 2, 4, or 8")
    rng = XorShift64(seed ^ (modulus * 0x9E3779B97F4A7C15))
    prefix = "Z" if heldout else TRAIN_PREFIXES[seed % len(TRAIN_PREFIXES)]
    width = max(2, len(str(entity_count - 1)))
    symbols = [f"{prefix}{index:0{width}d}" for index in range(premises + 1)]
    shifts: list[int] = []
    statements: list[str] = []
    for index in range(premises):
        shift = rng.integer(0, modulus)
        shifts.append(shift)
        left, right = symbols[index], symbols[index + 1]
        # Reversing an edge also inverts its public transformation.
        if rng.coin():
            statements.append(f"{right} IS SHIFT{(-shift) % modulus} {left}")
        else:
            statements.append(f"{left} IS SHIFT{shift} {right}")
    truth = sum(shifts) % modulus
    answer = rng.coin()
    proposed = (truth if answer else
                (truth + 1 + rng.integer(0, modulus - 1)) % modulus)
    if rng.coin():
        subject, obj, query_shift = symbols[-1], symbols[0], (-proposed) % modulus
    else:
        subject, obj, query_shift = symbols[0], symbols[-1], proposed
    for index in range(len(statements) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        statements[index], statements[other] = statements[other], statements[index]
    texts = tuple(statements) + (f"{subject} IS SHIFT{query_shift} {obj}",)
    frames = np.stack([
        render_public_card(
            text, index + 1, len(texts),
            (seed * 0x9E3779B1 + index * 0x85EBCA77) & 0x7FFF_FFFF,
            is_final=index == len(texts) - 1,
        ) for index, text in enumerate(texts)
    ])
    pcm = np.stack([render_public_audio(index, PCM_SAMPLES)
                    for index in range(len(texts))])
    actions = np.full(len(texts), int(Action.NEXT), dtype=np.int64)
    actions[-1] = int(Action.TRUE if answer else Action.FALSE)
    subjects, relations, objects = [], [], []
    for text in texts:
        text_subject, tail = text.split(" IS ", 1)
        relation, text_object = tail.rsplit(" ", 1)
        subjects.append(int(text_subject[1:]))
        relations.append(int(relation.removeprefix("SHIFT")))
        objects.append(int(text_object[1:]))
    return PublicEpisode(frames, pcm, actions,
                         np.asarray(subjects, dtype=np.int64),
                         np.asarray(relations, dtype=np.int64),
                         np.asarray(objects, dtype=np.int64), len(texts), seed)


class CyclicDataset(Dataset):
    def __init__(self, samples: int, premise_choices: tuple[int, ...], modulus: int, *,
                 start_seed: int = 0, heldout: bool = False):
        self.samples = samples
        self.premise_choices = premise_choices
        self.modulus = modulus
        self.start_seed = start_seed
        self.heldout = heldout

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PublicEpisode:
        seed = self.start_seed + index
        premises = self.premise_choices[seed % len(self.premise_choices)]
        return generate_cyclic_episode(seed, premises, self.modulus,
                                       heldout=self.heldout)
