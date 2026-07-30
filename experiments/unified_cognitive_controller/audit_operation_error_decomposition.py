"""Read-only decomposition of conditional-operation errors.

This diagnostic never trains a model.  It compares the conditional controller
with its frozen numerosity parent on exactly the same clean stimulus frames,
then separates errors by public operation, count-order relation, event
position, recurrent transition, and independent-event replay.  Verifier-private
metadata is used only in the emitted report and never enters either controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .environment import ACTIONS, NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController
from .train import rollout


def _load(
        path: Path, device: torch.device,
        ) -> tuple[dict[str, object], UnifiedCognitiveController]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError(f"{path} is not a unified controller checkpoint")
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"{path} lacks model_configuration")
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _group_accuracy(
        correct: torch.Tensor, groups: torch.Tensor,
        ) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for value in groups.unique(sorted=True).tolist():
        selected = groups == value
        result[str(int(value))] = {
            "count": int(selected.sum()),
            "accuracy": float(correct[selected].float().mean()),
        }
    return result


def _cross_accuracy(
        correct: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
        ) -> dict[str, dict[str, float | int]]:
    return {
        f"{int(first_value)}:{int(second_value)}": {
            "count": int(selected.sum()),
            "accuracy": float(correct[selected].float().mean()),
        }
        for first_value in first.unique(sorted=True).tolist()
        for second_value in second.unique(sorted=True).tolist()
        if bool((selected := (
            (first == first_value) & (second == second_value))).any())
    }


@torch.no_grad()
def _independent_logits(
        model: UnifiedCognitiveController, batch: CognitiveLifetimeBatch,
        ) -> torch.Tensor:
    events = batch.batch_size * batch.trials
    device = batch.frames.device
    state = model.initial_state(events, device=device, dtype=batch.frames.dtype)
    null = torch.full(
        (events,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(events, device=device)
    assert batch.prestimulus_frames is not None
    _, state = model.step(
        batch.prestimulus_frames.flatten(0, 1), state, null, zeros, zeros)
    output, _ = model.step(
        batch.frames.flatten(0, 1), state, null, zeros, zeros)
    return output.logits.reshape(batch.batch_size, batch.trials, ACTIONS)


def _operation_batch(
        *, count: int, seed: int, device: torch.device,
        reverse_operations: bool = False,
        ) -> CognitiveLifetimeBatch:
    return generate_lifetimes(
        count, 6, seed=seed, heldout=True,
        task="visible_pair_numerosity_operation",
        reverse_operations=reverse_operations,
        numerosity_appearance_blend=0.248,
        operation_cue_prestimulus=True,
        support_trials=1,
        device=device)


@torch.no_grad()
def audit(
        candidate: UnifiedCognitiveController,
        parent: UnifiedCognitiveController, *,
        count: int, seed: int, device: torch.device,
        ) -> dict[str, object]:
    normal = _operation_batch(count=count, seed=seed, device=device)
    reversed_batch = _operation_batch(
        count=count, seed=seed, device=device, reverse_operations=True)
    assert normal.context_ids is not None
    assert reversed_batch.context_ids is not None
    assert torch.equal(normal.frames, reversed_batch.frames)
    assert torch.equal(
        normal.correct_actions, 1 - reversed_batch.correct_actions)

    operation = normal.context_ids
    relation = normal.correct_actions ^ operation
    recurrent = rollout(
        candidate, normal, sample_actions=False, feedback_trials=1)
    reversed_recurrent = rollout(
        candidate, reversed_batch, sample_actions=False, feedback_trials=1)
    independent_logits = _independent_logits(candidate, normal)

    recurrent_actions = recurrent["actions"]
    reversed_actions = reversed_recurrent["actions"]
    independent_actions = independent_logits.argmax(-1)
    recurrent_correct = recurrent_actions == normal.correct_actions
    reversed_correct = (
        reversed_actions == reversed_batch.correct_actions)
    independent_correct = independent_actions == normal.correct_actions

    # The parent receives the exact same clean count frames, with no operation
    # cue or extra timestep.  This isolates how much of the conditional error
    # is inherited from the already learned larger-count relation.
    parent_batch = CognitiveLifetimeBatch(
        frames=normal.frames,
        correct_actions=relation,
        stimulus_identities=normal.stimulus_identities,
        rule_bits=normal.rule_bits,
        seeds=normal.seeds,
        context_ids=relation,
        prestimulus_frames=None)
    parent_result = rollout(
        parent, parent_batch, sample_actions=False, feedback_trials=1)
    parent_correct = parent_result["actions"] == relation

    trial = torch.arange(6, device=device).expand(count, -1)
    previous_operation = torch.cat(
        [torch.full_like(operation[:, :1], -1), operation[:, :-1]], dim=1)
    operation_transition = torch.where(
        trial == 0, torch.full_like(trial, 2),
        (operation != previous_operation).long())
    margin = (
        recurrent["logits"].softmax(-1).amax(-1)
        - recurrent["logits"].softmax(-1).amin(-1))
    counterfactual_flip = recurrent_actions != reversed_actions
    sequential_independent_disagreement = (
        recurrent_actions != independent_actions)

    query = trial >= 1
    parent_pass = parent_correct & query
    parent_fail = (~parent_correct) & query
    report: dict[str, object] = {
        "schema": "operation-error-decomposition-v1",
        "candidate_accuracy": {
            "all": float(recurrent_correct.float().mean()),
            "query_suffix": float(recurrent_correct[query].float().mean()),
            "independent_all": float(independent_correct.float().mean()),
            "independent_query_suffix":
                float(independent_correct[query].float().mean()),
        },
        "parent_relation_accuracy": {
            "all": float(parent_correct.float().mean()),
            "query_suffix": float(parent_correct[query].float().mean()),
            "query_by_relation":
                _group_accuracy(parent_correct[query], relation[query]),
        },
        "candidate_query_accuracy_conditioned_on_parent": {
            "parent_correct": float(
                recurrent_correct[parent_pass].float().mean()),
            "parent_wrong": float(
                recurrent_correct[parent_fail].float().mean()),
            "parent_wrong_fraction": float(parent_fail.float().mean()),
            "candidate_errors_with_parent_wrong_fraction": float(
                ((~recurrent_correct) & parent_fail).sum()
                / ((~recurrent_correct) & query).sum().clamp_min(1)),
        },
        "query_accuracy_by_operation":
            _group_accuracy(recurrent_correct[query], operation[query]),
        "query_accuracy_by_relation":
            _group_accuracy(recurrent_correct[query], relation[query]),
        "query_accuracy_by_correct_action": _group_accuracy(
            recurrent_correct[query], normal.correct_actions[query]),
        "query_prediction_rate_by_operation_relation": {
            key: {
                **value,
                "predicted_action_one_rate": float(
                    recurrent_actions[query][
                        (operation[query] == int(key.split(":")[0]))
                        & (relation[query] == int(key.split(":")[1]))]
                    .float().mean()),
            }
            for key, value in _cross_accuracy(
                recurrent_correct[query],
                operation[query], relation[query]).items()
        },
        "accuracy_by_trial":
            _group_accuracy(recurrent_correct, trial),
        "query_accuracy_by_operation_transition":
            _group_accuracy(
                recurrent_correct[query], operation_transition[query]),
        "counterfactual": {
            "normal_accuracy": float(recurrent_correct[query].float().mean()),
            "reversed_accuracy":
                float(reversed_correct[query].float().mean()),
            "prediction_flip_rate":
                float(counterfactual_flip[query].float().mean()),
            "both_directions_correct": float(
                (recurrent_correct & reversed_correct)[query].float().mean()),
            "neither_direction_correct": float(
                ((~recurrent_correct) & (~reversed_correct))[query]
                .float().mean()),
        },
        "history_dependence": {
            "prediction_disagreement_rate": float(
                sequential_independent_disagreement[query].float().mean()),
            "sequential_only_correct": float(
                (recurrent_correct & ~independent_correct)[query]
                .float().mean()),
            "independent_only_correct": float(
                (~recurrent_correct & independent_correct)[query]
                .float().mean()),
        },
        "confidence": {
            "mean_margin_correct": float(
                margin[recurrent_correct & query].mean()),
            "mean_margin_wrong": float(
                margin[(~recurrent_correct) & query].mean()),
            "wrong_above_0_8_margin_fraction": float(
                (margin[(~recurrent_correct) & query] >= 0.8)
                .float().mean()),
        },
        "accounting": {
            "logical_lifetimes": count,
            "sensory_frames": count * 6 * 2,
            "verifier_outcomes_used_for_training": 0,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=92_025_301)
    parser.add_argument(
        "--device", default=(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and even")
    device = torch.device(args.device)
    _, candidate = _load(args.candidate, device)
    _, parent = _load(args.parent, device)
    result = audit(
        candidate, parent, count=args.count, seed=args.seed, device=device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
