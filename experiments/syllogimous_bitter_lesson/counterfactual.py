from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.syllogimous_latent_agent.data import (
    RELATION_TO_ID,
    PublicEpisode,
    balanced_question,
    generate_public_episode,
    render_public_card,
    visible_texts,
)
from experiments.syllogimous_realtime.environment import Action, RELATIONS

from .model import BitterLessonAgent


RELATION_MATE = {left: right for pair in RELATIONS for left, right in (pair, pair[::-1])}


def counterfactual_pair(seed: int, premises: int, *, heldout: bool = True,
                        entity_count: int = 128,
                        randomize_rendering: bool = True) -> tuple[PublicEpisode, PublicEpisode]:
    """Change only the visible conclusion relation and invert its verifier answer."""
    original = generate_public_episode(seed, premises, heldout=heldout, final=True,
                                       entity_count=entity_count,
                                       randomize_rendering=randomize_rendering)
    question = balanced_question(seed, premises, heldout=heldout, final=True,
                                 entity_count=entity_count)
    texts = visible_texts(question, heldout=heldout)
    subject, tail = texts[-1].split(" IS ", 1)
    relation, obj = tail.rsplit(" ", 1)
    flipped_relation = RELATION_MATE[relation]
    flipped_text = f"{subject} IS {flipped_relation} {obj}"
    style_seed = ((seed * 0x9E3779B1 + premises * 0x85EBCA77) & 0x7FFF_FFFF
                  if randomize_rendering else premises)
    frames = original.frames.copy()
    frames[-1] = render_public_card(flipped_text, premises + 1, premises + 1,
                                    style_seed, is_final=True)
    actions = original.actions.copy()
    actions[-1] = int(Action.FALSE if actions[-1] == int(Action.TRUE) else Action.TRUE)
    relations = original.relations.copy()
    relations[-1] = RELATION_TO_ID[flipped_relation]
    counterfactual = PublicEpisode(frames, original.pcm.copy(), actions,
                                   original.subjects.copy(), relations,
                                   original.objects.copy(), original.length, original.seed)
    return original, counterfactual


def _public_tensors(episodes: list[PublicEpisode], device: torch.device) -> tuple:
    from experiments.syllogimous_latent_agent.data import collate_episodes

    batch = collate_episodes(episodes)
    return (batch["frames"].to(device), batch["pcm"].to(device),
            batch["mask"].to(device), batch["actions"].to(device))


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--premises", default="2,4,8,16,64")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload["metadata"]["config"]
    model = BitterLessonAgent(**config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    lengths = tuple(map(int, args.premises.split(",")))
    original_right = counterfactual_right = both_right = predictions_flipped = 0
    by_length: dict[int, dict[str, int]] = {}
    for start in range(0, args.samples, args.batch_size):
        originals, counterfactuals = [], []
        batch_lengths = []
        for index in range(start, min(args.samples, start + args.batch_size)):
            premises = lengths[index % len(lengths)]
            original, counterfactual = counterfactual_pair(200_000 + index, premises)
            originals.append(original)
            counterfactuals.append(counterfactual)
            batch_lengths.append(premises)
        original_inputs = _public_tensors(originals, device)
        counterfactual_inputs = _public_tensors(counterfactuals, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            original_output = model(*original_inputs[:3])
            counterfactual_output = model(*counterfactual_inputs[:3])
        original_predictions = original_output.answer_logits[:, -1].argmax(-1)
        counterfactual_predictions = counterfactual_output.answer_logits[:, -1].argmax(-1)
        rows = torch.arange(len(originals), device=device)
        final_indices = original_inputs[2].sum(1) - 1
        original_targets = original_inputs[3][rows, final_indices]
        counterfactual_targets = counterfactual_inputs[3][rows, final_indices]
        original_matches = original_predictions == original_targets
        counterfactual_matches = counterfactual_predictions == counterfactual_targets
        flipped = original_predictions != counterfactual_predictions
        original_right += int(original_matches.sum())
        counterfactual_right += int(counterfactual_matches.sum())
        both_right += int((original_matches & counterfactual_matches).sum())
        predictions_flipped += int(flipped.sum())
        for length, left, right, pair, changed in zip(
                batch_lengths, original_matches.tolist(), counterfactual_matches.tolist(),
                (original_matches & counterfactual_matches).tolist(), flipped.tolist()):
            bucket = by_length.setdefault(length, {"episodes": 0, "original_right": 0,
                                                   "counterfactual_right": 0,
                                                   "both_right": 0,
                                                   "prediction_flipped": 0})
            bucket["episodes"] += 1
            bucket["original_right"] += int(left)
            bucket["counterfactual_right"] += int(right)
            bucket["both_right"] += int(pair)
            bucket["prediction_flipped"] += int(changed)
    result = {
        "episodes": args.samples,
        "original_accuracy": original_right / args.samples,
        "counterfactual_accuracy": counterfactual_right / args.samples,
        "paired_both_correct": both_right / args.samples,
        "prediction_flip_rate": predictions_flipped / args.samples,
        "by_premises": {
            str(length): {key: (value / values["episodes"] if key != "episodes" else value)
                          for key, value in values.items()}
            for length, values in sorted(by_length.items())
        },
    }
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

