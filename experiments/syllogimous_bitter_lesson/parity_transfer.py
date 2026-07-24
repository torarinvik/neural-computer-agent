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


def generate_parity_episode(seed: int, premises: int, *, heldout: bool = False,
                            entity_count: int = 128) -> PublicEpisode:
    """Compose SAME/FLIP constraints along a shuffled path."""
    if premises < 2 or premises >= entity_count:
        raise ValueError("premises must be between 2 and entity_count - 1")
    rng = XorShift64(seed)
    prefix = "Z" if heldout else TRAIN_PREFIXES[seed % len(TRAIN_PREFIXES)]
    width = max(2, len(str(entity_count - 1)))
    symbols = [f"{prefix}{index:0{width}d}" for index in range(premises + 1)]
    edge_parity = []
    statements = []
    for index in range(premises):
        flipped = rng.coin()
        edge_parity.append(flipped)
        relation = "FLIP" if flipped else "SAME"
        left, right = symbols[index], symbols[index + 1]
        statements.append(f"{right} IS {relation} {left}" if rng.coin()
                          else f"{left} IS {relation} {right}")
    truth_is_flip = bool(sum(edge_parity) % 2)
    proposed_is_flip = rng.coin()
    answer = proposed_is_flip == truth_is_flip
    conclusion_relation = "FLIP" if proposed_is_flip else "SAME"
    subject, obj = ((symbols[-1], symbols[0]) if rng.coin()
                    else (symbols[0], symbols[-1]))
    for index in range(len(statements) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        statements[index], statements[other] = statements[other], statements[index]
    texts = tuple(statements) + (f"{subject} IS {conclusion_relation} {obj}",)
    frames = np.stack([
        render_public_card(
            text, index + 1, len(texts),
            ((seed * 0x9E3779B1 + index * 0x85EBCA77) & 0x7FFF_FFFF),
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
        relations.append(1 if relation == "FLIP" else 0)
        objects.append(int(text_object[1:]))
    return PublicEpisode(frames, pcm, actions,
                         np.asarray(subjects, dtype=np.int64),
                         np.asarray(relations, dtype=np.int64),
                         np.asarray(objects, dtype=np.int64), len(texts), seed)


class ParityDataset(Dataset):
    def __init__(self, samples: int, premise_choices: tuple[int, ...], *,
                 start_seed: int = 0, heldout: bool = False):
        self.samples = samples
        self.premise_choices = premise_choices
        self.start_seed = start_seed
        self.heldout = heldout

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PublicEpisode:
        seed = self.start_seed + index
        premises = self.premise_choices[seed % len(self.premise_choices)]
        return generate_parity_episode(seed, premises, heldout=self.heldout)

