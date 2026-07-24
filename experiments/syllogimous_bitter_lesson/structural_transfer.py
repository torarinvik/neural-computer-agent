from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.syllogimous_latent_agent.data import (
    PCM_SAMPLES,
    RELATION_TO_ID,
    PublicEpisode,
    EpisodeDataset,
    TRAIN_PREFIXES,
    collate_episodes,
    render_public_audio,
    render_public_card,
)
from experiments.syllogimous_realtime.environment import Action, RELATIONS, XorShift64

from .model import BitterLessonAgent


def _statement(left: str, forward: str, right: str, rng: XorShift64) -> str:
    reverse = dict(RELATIONS)[forward]
    return (f"{left} IS {forward} {right}" if rng.coin()
            else f"{right} IS {reverse} {left}")


def generate_branched_episode(seed: int, total_premises: int, relevant_depth: int,
                              *, entity_count: int = 128,
                              heldout: bool = True) -> PublicEpisode:
    """A relevant path hidden among disconnected, mixed-relation distractors."""
    if relevant_depth < 2 or relevant_depth > total_premises:
        raise ValueError("relevant depth must be between 2 and total premises")
    if total_premises >= entity_count:
        raise ValueError("total premises must be smaller than entity count")
    rng = XorShift64(seed)
    width = max(2, len(str(entity_count - 1)))
    prefix = "Z" if heldout else TRAIN_PREFIXES[seed % len(TRAIN_PREFIXES)]
    symbols = [f"{prefix}{index:0{width}d}" for index in range(entity_count)]
    main_pair = RELATIONS[rng.integer(0, len(RELATIONS))]
    forward, reverse = main_pair
    premises = [_statement(symbols[i], forward, symbols[i + 1], rng)
                for i in range(relevant_depth)]

    # Distractors occupy a disjoint entity chain and deliberately mix relation
    # families. They are public propositions, but none connects query endpoints.
    distractor_start = relevant_depth + 1
    distractor_symbols = symbols[distractor_start:]
    for index in range(total_premises - relevant_depth):
        left = distractor_symbols[index % (len(distractor_symbols) - 1)]
        right = distractor_symbols[(index + 1) % len(distractor_symbols)]
        distractor_forward, _ = RELATIONS[rng.integer(0, len(RELATIONS))]
        premises.append(_statement(left, distractor_forward, right, rng))
    for index in range(len(premises) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        premises[index], premises[other] = premises[other], premises[index]

    answer = rng.coin()
    swap_endpoints = rng.coin()
    if swap_endpoints:
        subject, obj = symbols[relevant_depth], symbols[0]
        truthful, false_relation = reverse, forward
    else:
        subject, obj = symbols[0], symbols[relevant_depth]
        truthful, false_relation = forward, reverse
    conclusion_relation = truthful if answer else false_relation
    texts = tuple(premises) + (f"{subject} IS {conclusion_relation} {obj}",)
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
        relations.append(RELATION_TO_ID[relation])
        objects.append(int(text_object[1:]))
    return PublicEpisode(frames, pcm, actions,
                         np.asarray(subjects, dtype=np.int64),
                         np.asarray(relations, dtype=np.int64),
                         np.asarray(objects, dtype=np.int64), len(texts), seed)


class MixedStructuralDataset(Dataset):
    """Half ordinary chains, half branched/distractor episodes."""

    def __init__(self, samples: int, chain_premises: tuple[int, ...],
                 branched_configs: tuple[tuple[int, int], ...]):
        self.samples = samples
        self.chains = EpisodeDataset((samples + 1) // 2,
                                     premise_choices=chain_premises,
                                     entity_count=128, randomize_rendering=True)
        self.branched_configs = branched_configs

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> PublicEpisode:
        if index % 2 == 0:
            return self.chains[index // 2]
        branched_index = index // 2
        total, depth = self.branched_configs[branched_index % len(self.branched_configs)]
        return generate_branched_episode(400_000 + branched_index, total, depth,
                                         heldout=False)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--configs", default="8:2,16:4,32:4,64:4,16:8,64:8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configs = tuple(tuple(map(int, item.split(":"))) for item in args.configs.split(","))
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BitterLessonAgent(**payload["metadata"]["config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    right = total = 0
    by_config: dict[str, list[int]] = {}
    for start in range(0, args.samples, args.batch_size):
        items, item_configs = [], []
        for index in range(start, min(args.samples, start + args.batch_size)):
            total_premises, depth = configs[index % len(configs)]
            items.append(generate_branched_episode(300_000 + index, total_premises, depth))
            item_configs.append((total_premises, depth))
        batch = collate_episodes(items)
        frames = batch["frames"].to(device)
        pcm = batch["pcm"].to(device)
        mask = batch["mask"].to(device)
        actions = batch["actions"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(frames, pcm, mask)
        rows = torch.arange(len(items), device=device)
        final_indices = mask.sum(1) - 1
        matches = output.answer_logits[:, -1].argmax(-1) == actions[rows, final_indices]
        right += int(matches.sum())
        total += len(items)
        for config, match in zip(item_configs, matches.tolist()):
            key = f"total_{config[0]}_depth_{config[1]}"
            bucket = by_config.setdefault(key, [0, 0])
            bucket[0] += int(match)
            bucket[1] += 1
    result = {"episodes": total, "accuracy": right / total,
              "accuracy_by_config": {key: values[0] / values[1]
                                     for key, values in sorted(by_config.items())}}
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
