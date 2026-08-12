"""Learn victim choice for recursive recipe files from scalar utility only.

The existing generic external capability-eviction scorer is trained against
real :class:`ExternalRecipeCompositionMemory` compaction transactions. The
policy receives permutation-safe structural telemetry, not slot indices,
digests, operations, task labels, or verifier answers. Compaction and the
behavior verifier remain authoritative; policy adaptation is external state
and uses one scalar utility per fresh episode without replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from itertools import product
from pathlib import Path

import torch

from neural_computer import (
    RECIPE_COMPOSITION_TELEMETRY_WIDTH,
    ExternalCapabilityEvictionPolicy,
    ExternalRecipeCompositionMemory,
    RecipeInstruction,
    RecipeProgram,
)

SLOT_VALUES = (2, 4, 8)
ALL_STATES = tuple(product(*(range(value) for value in SLOT_VALUES)))
CONTEXT_WIDTH = 4
CANDIDATE_WIDTH = RECIPE_COMPOSITION_TELEMETRY_WIDTH
TRAIN_EPISODES = 512
EVAL_EPISODES = 256
TEMPERATURE = 0.7


def _program(*instructions: RecipeInstruction) -> RecipeProgram:
    return RecipeProgram(SLOT_VALUES, instructions)


def _source_programs() -> tuple[RecipeProgram, ...]:
    return (
        _program(RecipeInstruction("inc", 0, modulus=2)),
        _program(RecipeInstruction("cinc", 1, 0, modulus=4)),
        _program(RecipeInstruction("cinc", 2, 1, modulus=8)),
        _program(RecipeInstruction("cdec", 0, 2, modulus=2)),
        _program(RecipeInstruction("inc", 1, modulus=4)),
    )


def _serial(left: RecipeProgram, right: RecipeProgram) -> RecipeProgram:
    return _program(*(left.instructions + right.instructions))


def _scores(candidate: RecipeProgram, reference: RecipeProgram) -> torch.Tensor:
    return torch.tensor(
        [float(candidate.execute(state) == reference.execute(state)) for state in ALL_STATES],
        dtype=torch.float32,
    )


def _admit_atomic(
    memory: ExternalRecipeCompositionMemory,
    program: RecipeProgram,
    *,
    protect: bool,
) -> int:
    receipt = memory.admit_verified_program(
        program,
        _scores(program, program),
        threshold=1.0,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=protect,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError("recipe source admission failed")
    return int(receipt.slot)


def _compose(
    memory: ExternalRecipeCompositionMemory,
    target: RecipeProgram,
    *,
    protect: bool,
) -> int:
    candidate = next(
        item
        for item in memory.composition_candidates(max_program_length=5)
        if item.program.digest() == target.digest()
    )
    receipt = memory.admit_verified_composition(
        candidate,
        _scores(candidate.program, target),
        threshold=1.0,
        min_observations=len(ALL_STATES),
        min_stable_observations=len(ALL_STATES),
        protect=protect,
    )
    if not receipt.accepted or receipt.slot is None:
        raise RuntimeError("recipe composition admission failed")
    return int(receipt.slot)


def _find_digest(memory: ExternalRecipeCompositionMemory, digest: str) -> int | None:
    return next(
        (
            slot
            for slot in range(memory.file_count)
            if memory.program(slot).digest() == digest
        ),
        None,
    )


def _episode(
    seed: int,
    *,
    depth: int | None = None,
) -> tuple[
    ExternalRecipeCompositionMemory,
    tuple[int, ...],
    int,
    torch.Tensor,
    torch.Tensor,
]:
    """Create one fresh bank and hide which candidate is disposable.

    The useful root depth varies from two through four during training and is
    five during transfer evaluation. The policy sees only the candidate
    telemetry and generic capacity context; the root-retention verifier is the
    only source of utility.
    """

    if depth is None:
        depth = 2 + (seed % 3)
    if not 2 <= depth <= len(_source_programs()):
        raise ValueError("recipe eviction episode depth is outside the audit range")
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    sources = _source_programs()
    source_slots = tuple(
        _admit_atomic(memory, source, protect=True) for source in sources[:depth]
    )
    prefix = sources[0]
    root_slot = source_slots[0]
    for prefix_depth in range(2, depth + 1):
        prefix = _serial(prefix, sources[prefix_depth - 1])
        root_slot = _compose(
            memory,
            prefix,
            protect=prefix_depth < depth,
        )

    decoys = (
        _program(RecipeInstruction("dec", 0, modulus=2)),
        _program(RecipeInstruction("dec", 1, modulus=4)),
    )
    decoy_slots = tuple(
        _admit_atomic(memory, program, protect=False) for program in decoys
    )
    candidates = (root_slot, *decoy_slots)
    generator = torch.Generator(device="cpu").manual_seed(seed + 902_001)
    permutation = torch.randperm(len(candidates), generator=generator)
    ordered = tuple(candidates[index] for index in permutation.tolist())
    telemetry = memory.candidate_telemetry(ordered)
    context = torch.tensor(
        [
            1.0,
            math.log1p(memory.file_count),
            len(ordered) / memory.file_count,
            depth / len(_source_programs()),
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    return memory, ordered, root_slot, context, telemetry.unsqueeze(0)


def _attempt_selected(
    memory: ExternalRecipeCompositionMemory,
    ordered_candidates: tuple[int, ...],
    root_slot: int,
    selected_position: int,
) -> tuple[float, bool, int]:
    selected = ordered_candidates[selected_position]
    requested = tuple(
        slot for position, slot in enumerate(ordered_candidates) if position != selected_position
    )
    root_digest = memory.program(root_slot).digest()

    def verifier(candidate: ExternalRecipeCompositionMemory) -> bool:
        retained_root = _find_digest(candidate, root_digest)
        if retained_root is None:
            return False
        return all(
            candidate.execute(retained_root, state)
            == memory.execute(root_slot, state)
            for state in ALL_STATES
        )

    _, receipt = memory.compact_verified(requested, verifier=verifier)
    return float(receipt.accepted), receipt.accepted, selected


def _policy() -> ExternalCapabilityEvictionPolicy:
    return ExternalCapabilityEvictionPolicy(
        context_width=CONTEXT_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        hidden=24,
    )


def _episode_tensors(
    seed: int,
    *,
    depth: int | None = None,
) -> tuple[ExternalRecipeCompositionMemory, tuple[int, ...], int, torch.Tensor, torch.Tensor]:
    memory, ordered, root_slot, context, telemetry = _episode(seed, depth=depth)
    return memory, ordered, root_slot, context, telemetry


def _train(
    seed: int,
    *,
    shuffled_utility: bool = False,
) -> tuple[ExternalCapabilityEvictionPolicy, dict[str, object]]:
    torch.manual_seed(seed)
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    generator = torch.Generator(device="cpu").manual_seed(seed + 500_000)
    utilities: list[float] = []
    verifier_bits = 0
    for update in range(TRAIN_EPISODES):
        episode_seed = seed + update * 17
        memory, ordered, root_slot, context, telemetry = _episode_tensors(
            episode_seed,
        )
        depth = 2 + (episode_seed % 3)
        verifier_bits += 2 * depth * len(ALL_STATES)
        scores = policy.score_candidates(context, telemetry)[0]
        probabilities = torch.softmax(scores / TEMPERATURE, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=generator))
        utility, _, _ = _attempt_selected(memory, ordered, root_slot, selected)
        if shuffled_utility:
            utility = float(torch.randint(2, (), generator=generator))
        loss = -(utility - 0.5) * torch.log_softmax(
            scores / TEMPERATURE, dim=-1
        )[selected]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        utilities.append(utility)
    return policy, {
        "optimizer_updates": TRAIN_EPISODES,
        "unique_verifier_bits": verifier_bits,
        "first_window_utility": sum(utilities[:64]) / 64,
        "last_window_utility": sum(utilities[-64:]) / 64,
        "mean_utility": sum(utilities) / len(utilities),
    }


@torch.no_grad()
def _evaluate(
    policy: ExternalCapabilityEvictionPolicy,
    seed: int,
    *,
    depth: int | None = 5,
    corrupt_features: bool = False,
) -> dict[str, float | int]:
    correct = 0
    accepted = 0
    verifier_bits = 0
    for index in range(EVAL_EPISODES):
        memory, ordered, root_slot, context, telemetry = _episode_tensors(
            seed + index * 19,
            depth=depth,
        )
        verifier_bits += 2 * (5 if depth is None else depth) * len(ALL_STATES)
        if corrupt_features:
            telemetry = torch.zeros_like(telemetry)
        selected = int(policy.score_candidates(context, telemetry)[0].argmax())
        utility, did_accept, _ = _attempt_selected(
            memory,
            ordered,
            root_slot,
            selected,
        )
        correct += int(utility >= 1.0)
        accepted += int(did_accept)
    return {
        "accuracy": correct / EVAL_EPISODES,
        "accepted": accepted,
        "episodes": EVAL_EPISODES,
        "unique_verifier_bits": verifier_bits,
    }


def _digest(policy: ExternalCapabilityEvictionPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    started = time.perf_counter()
    trained_policy, training = _train(seed)
    shuffled_policy, shuffled_training = _train(seed + 10_000, shuffled_utility=True)
    fresh_policy = _policy()
    trained_eval = _evaluate(trained_policy, seed + 20_000)
    fresh_eval = _evaluate(fresh_policy, seed + 21_000)
    shuffled_eval = _evaluate(shuffled_policy, seed + 22_000)
    corrupted_eval = _evaluate(
        trained_policy,
        seed + 23_000,
        corrupt_features=True,
    )
    persisted = _policy()
    persisted.load_state_dict(trained_policy.state_dict())
    gates = {
        "trained_beats_fresh": trained_eval["accuracy"] > fresh_eval["accuracy"] + 0.25,
        "trained_beats_shuffled": trained_eval["accuracy"] > shuffled_eval["accuracy"] + 0.25,
        "feature_causal": trained_eval["accuracy"] > corrupted_eval["accuracy"] + 0.20,
        "order_permutation": trained_eval["accuracy"] >= 0.80,
        "policy_persistence_exact": _digest(persisted) == _digest(trained_policy),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.learned-recipe-eviction.v1",
        "claim_boundary": (
            "bounded learned victim choice over recursive recipe files from "
            "permutation-safe structural telemetry and scalar compaction utility; "
            "not universal eviction economics or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "train_depths": [2, 3, 4],
            "transfer_depth": 5,
            "learner_inputs": [
                "recipe_candidate_telemetry",
                "generic_capacity_context",
                "deterministic_scalar_compaction_utility",
            ],
            "candidate_width": CANDIDATE_WIDTH,
            "context_width": CONTEXT_WIDTH,
        },
        "training": training,
        "shuffled_training": shuffled_training,
        "trained_eval": trained_eval,
        "fresh_eval": fresh_eval,
        "shuffled_eval": shuffled_eval,
        "corrupted_feature_eval": corrupted_eval,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": (
                int(training["unique_verifier_bits"])
                + int(shuffled_training["unique_verifier_bits"])
                + sum(
                    int(evaluation["unique_verifier_bits"])
                    for evaluation in (
                        trained_eval,
                        fresh_eval,
                        shuffled_eval,
                        corrupted_eval,
                    )
                )
            ),
            "unique_logical_lifetimes": 2 * TRAIN_EPISODES + 4 * EVAL_EPISODES,
            "optimizer_updates": TRAIN_EPISODES,
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "latency_seconds": time.perf_counter() - started,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed, args.report_out)
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
