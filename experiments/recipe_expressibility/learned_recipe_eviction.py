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
    RecipeProgramCompositionCandidate,
    RecipeProgramCompositionFactors,
)

SLOT_VALUES = (2, 4, 8)
ALL_STATES = tuple(product(*(range(value) for value in SLOT_VALUES)))
DEPTH_PROFILES = {
    2: ((2, 3), (3, 4)),
    3: ((2, 3, 4), (3, 4, 5)),
    4: ((2, 3, 4, 5), (3, 4, 5, 6)),
}
TRAIN_DEPTHS, TRANSFER_DEPTHS = DEPTH_PROFILES[2]
CANDIDATE_COUNT = 2
CONTEXT_WIDTH = 4
CANDIDATE_WIDTH = RECIPE_COMPOSITION_TELEMETRY_WIDTH
TRAIN_EPISODES = 512
EVAL_EPISODES = 256
PROBE_INTERVAL = 64
PROBE_EPISODES = 32
STABLE_ACCURACY_THRESHOLD = 0.90
CREDIT_MODES = ("sampled", "counterfactual")
CAUSAL_MARGIN = 0.20
TEMPERATURE = 0.7


def configure_profile(candidate_count: int) -> None:
    """Select a bounded candidate-count rung for the standalone audit."""

    global CANDIDATE_COUNT, TRAIN_DEPTHS, TRANSFER_DEPTHS
    try:
        train_depths, transfer_depths = DEPTH_PROFILES[int(candidate_count)]
    except KeyError as error:
        raise ValueError("candidate-count must be one of 2, 3, or 4") from error
    CANDIDATE_COUNT = int(candidate_count)
    TRAIN_DEPTHS = train_depths
    TRANSFER_DEPTHS = transfer_depths


def _program(*instructions: RecipeInstruction) -> RecipeProgram:
    return RecipeProgram(SLOT_VALUES, instructions)


def _source_programs() -> tuple[RecipeProgram, ...]:
    return (
        _program(RecipeInstruction("inc", 0, modulus=2)),
        _program(RecipeInstruction("cinc", 1, 0, modulus=4)),
        _program(RecipeInstruction("cinc", 2, 1, modulus=8)),
        _program(RecipeInstruction("cdec", 0, 2, modulus=2)),
        _program(RecipeInstruction("inc", 1, modulus=4)),
        _program(RecipeInstruction("dec", 2, modulus=8)),
        _program(RecipeInstruction("cinc", 0, 2, modulus=2)),
        _program(RecipeInstruction("cdec", 1, 2, modulus=4)),
        _program(RecipeInstruction("inc", 2, modulus=8)),
        _program(RecipeInstruction("dec", 0, modulus=2)),
        _program(RecipeInstruction("cdec", 2, 0, modulus=8)),
        _program(RecipeInstruction("cinc", 1, 2, modulus=4)),
        _program(RecipeInstruction("dec", 1, modulus=4)),
        _program(RecipeInstruction("cinc", 2, 0, modulus=8)),
        _program(RecipeInstruction("cdec", 0, 1, modulus=2)),
        _program(RecipeInstruction("cinc", 0, 1, modulus=2)),
        _program(RecipeInstruction("cdec", 1, 0, modulus=4)),
        _program(RecipeInstruction("cdec", 2, 1, modulus=8)),
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
    left_slot: int,
    right_slot: int,
    protect: bool,
) -> int:
    candidate = RecipeProgramCompositionCandidate(
        left_slot=left_slot,
        right_slot=right_slot,
        factors=RecipeProgramCompositionFactors(
            memory.program(left_slot).digest(),
            memory.program(right_slot).digest(),
            "append",
        ),
        program=target,
        structure=memory.composition_structure(left_slot, right_slot),
    )
    candidate.validate()
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
    candidate_depths: tuple[int, ...],
) -> tuple[
    ExternalRecipeCompositionMemory,
    tuple[int, ...],
    int,
    torch.Tensor,
    torch.Tensor,
]:
    """Create a fresh bank with a context-conditioned required root.

    Each candidate is an independent recursive file, so no candidate can be
    retained accidentally through another candidate's provenance closure. A
    generic capacity-pressure regime determines which depth rank is currently
    required. The policy sees the regime and structural telemetry, but not the
    required-root identity; scalar verifier utility is the only credit signal.
    """

    depths = tuple(int(depth) for depth in candidate_depths)
    if len(depths) != CANDIDATE_COUNT or len(set(depths)) != len(depths):
        raise ValueError("recipe eviction needs four distinct candidate depths")
    if any(depth < 2 or depth > 6 for depth in depths):
        raise ValueError("recipe eviction candidate depth is outside the audit range")
    memory = ExternalRecipeCompositionMemory(SLOT_VALUES)
    sources = _source_programs()
    roots: list[int] = []
    source_cursor = 0
    for depth in depths:
        if source_cursor + depth > len(sources):
            raise ValueError("recipe source bank is too small for candidate depths")
        chain_sources = sources[source_cursor : source_cursor + depth]
        source_cursor += depth
        source_slots = tuple(
            _admit_atomic(memory, source, protect=True) for source in chain_sources
        )
        prefix = chain_sources[0]
        root_slot = source_slots[0]
        for prefix_depth in range(2, depth + 1):
            prefix = _serial(prefix, chain_sources[prefix_depth - 1])
            root_slot = _compose(
                memory,
                prefix,
                left_slot=root_slot,
                right_slot=source_slots[prefix_depth - 1],
                protect=prefix_depth < depth,
            )
        roots.append(root_slot)

    pressure_regime = seed % CANDIDATE_COUNT
    pressure = pressure_regime / (CANDIDATE_COUNT - 1)
    active_rank = pressure_regime
    active_root = roots[active_rank]
    generator = torch.Generator(device="cpu").manual_seed(seed + 902_001)
    permutation = torch.randperm(len(roots), generator=generator)
    ordered = tuple(roots[index] for index in permutation.tolist())
    telemetry = memory.candidate_telemetry(ordered)
    context = torch.tensor(
        [
            1.0,
            pressure,
            math.log1p(memory.file_count),
            len(ordered) / memory.file_count,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    return memory, ordered, active_root, context, telemetry.unsqueeze(0)


def _attempt_selected(
    memory: ExternalRecipeCompositionMemory,
    ordered_candidates: tuple[int, ...],
    required_root_slot: int,
    selected_position: int,
) -> tuple[float, bool, int]:
    selected = ordered_candidates[selected_position]
    requested = tuple(
        slot for position, slot in enumerate(ordered_candidates) if position != selected_position
    )
    root_digest = memory.program(required_root_slot).digest()

    def verifier(candidate: ExternalRecipeCompositionMemory) -> bool:
        retained_root = _find_digest(candidate, root_digest)
        if retained_root is None:
            return False
        return all(
            candidate.execute(retained_root, state)
            == memory.execute(required_root_slot, state)
            for state in ALL_STATES
        )

    _, receipt = memory.compact_verified(requested, verifier=verifier)
    return float(receipt.accepted), receipt.accepted, selected


def _attempt_all(
    memory: ExternalRecipeCompositionMemory,
    ordered_candidates: tuple[int, ...],
    required_root_slot: int,
) -> torch.Tensor:
    """Collect one scalar verifier utility for every available victim choice."""

    return torch.tensor(
        [
            _attempt_selected(
                memory,
                ordered_candidates,
                required_root_slot,
                position,
            )[0]
            for position in range(len(ordered_candidates))
        ],
        dtype=torch.float32,
    )


def _policy() -> ExternalCapabilityEvictionPolicy:
    return ExternalCapabilityEvictionPolicy(
        context_width=CONTEXT_WIDTH,
        candidate_width=CANDIDATE_WIDTH,
        hidden=24,
    )


def _episode_tensors(
    seed: int,
    *,
    candidate_depths: tuple[int, ...],
) -> tuple[ExternalRecipeCompositionMemory, tuple[int, ...], int, torch.Tensor, torch.Tensor]:
    return _episode(seed, candidate_depths=candidate_depths)


def _verifier_bits_for_episode(
    candidate_depths: tuple[int, ...],
    *,
    candidate_evaluations: int = 1,
) -> int:
    if candidate_evaluations < 1:
        raise ValueError("candidate evaluations must be positive")
    admission_and_composition = 2 * sum(candidate_depths) - len(candidate_depths)
    return (admission_and_composition + candidate_evaluations) * len(ALL_STATES)


def _train(
    seed: int,
    *,
    candidate_depths: tuple[int, ...],
    credit_mode: str,
    shuffled_utility: bool = False,
) -> tuple[ExternalCapabilityEvictionPolicy, dict[str, object]]:
    if credit_mode not in CREDIT_MODES:
        raise ValueError(f"unknown maintenance credit mode: {credit_mode!r}")
    torch.manual_seed(seed)
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    generator = torch.Generator(device="cpu").manual_seed(seed + 500_000)
    utilities: list[float] = []
    probes: list[dict[str, float | int]] = []
    verifier_bits = 0
    for update in range(TRAIN_EPISODES):
        episode_seed = seed + update * 17
        memory, ordered, root_slot, context, telemetry = _episode_tensors(
            episode_seed,
            candidate_depths=candidate_depths,
        )
        scores = policy.score_candidates(context, telemetry)[0]
        probabilities = torch.softmax(scores / TEMPERATURE, dim=-1)
        if credit_mode == "counterfactual":
            utility_values = _attempt_all(memory, ordered, root_slot)
            verifier_bits += _verifier_bits_for_episode(
                candidate_depths,
                candidate_evaluations=len(ordered),
            )
            if shuffled_utility:
                utility_values = torch.randint(
                    2,
                    utility_values.shape,
                    generator=generator,
                    dtype=torch.int64,
                ).to(dtype=utility_values.dtype)
            expected_utility = float((probabilities.detach() * utility_values).sum())
            utility = expected_utility
            loss = -(probabilities * utility_values).sum()
        else:
            verifier_bits += _verifier_bits_for_episode(candidate_depths)
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
        if not shuffled_utility and (update + 1) % PROBE_INTERVAL == 0:
            probe = _evaluate(
                policy,
                seed + 700_000 + update,
                candidate_depths=candidate_depths,
                episodes=PROBE_EPISODES,
            )
            verifier_bits += int(probe["unique_verifier_bits"])
            probes.append(
                {
                    "update": update + 1,
                    "accuracy": float(probe["accuracy"]),
                    "verifier_bits": verifier_bits,
                }
            )
    stable_update: int | None = None
    stable_verifier_bits: int | None = None
    for index, probe in enumerate(probes):
        if float(probe["accuracy"]) >= STABLE_ACCURACY_THRESHOLD and all(
            float(later["accuracy"]) >= STABLE_ACCURACY_THRESHOLD
            for later in probes[index:]
        ):
            stable_update = int(probe["update"])
            stable_verifier_bits = int(probe["verifier_bits"])
            break
    return policy, {
        "optimizer_updates": TRAIN_EPISODES,
        "unique_verifier_bits": verifier_bits,
        "first_window_utility": sum(utilities[:64]) / 64,
        "last_window_utility": sum(utilities[-64:]) / 64,
        "mean_utility": sum(utilities) / len(utilities),
        "probe_history": probes,
        "stable_accuracy_threshold": STABLE_ACCURACY_THRESHOLD,
        "stable_update": stable_update,
        "stable_verifier_bits": stable_verifier_bits,
    }


@torch.no_grad()
def _evaluate(
    policy: ExternalCapabilityEvictionPolicy,
    seed: int,
    *,
    candidate_depths: tuple[int, ...],
    episodes: int | None = None,
    corrupt_features: bool = False,
) -> dict[str, float | int]:
    episode_count = EVAL_EPISODES if episodes is None else int(episodes)
    if episode_count < 1:
        raise ValueError("evaluation episodes must be positive")
    correct = 0
    accepted = 0
    verifier_bits = 0
    for index in range(episode_count):
        memory, ordered, root_slot, context, telemetry = _episode_tensors(
            seed + index * 19,
            candidate_depths=candidate_depths,
        )
        verifier_bits += _verifier_bits_for_episode(candidate_depths)
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
        "accuracy": correct / episode_count,
        "accepted": accepted,
        "episodes": episode_count,
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


def run(
    seed: int,
    report_out: Path,
    *,
    candidate_count: int | None = None,
    credit_mode: str = "sampled",
) -> dict[str, object]:
    if candidate_count is not None:
        configure_profile(candidate_count)
    started = time.perf_counter()
    trained_policy, training = _train(
        seed,
        candidate_depths=TRAIN_DEPTHS,
        credit_mode=credit_mode,
    )
    shuffled_policy, shuffled_training = _train(
        seed + 10_000,
        candidate_depths=TRAIN_DEPTHS,
        credit_mode=credit_mode,
        shuffled_utility=True,
    )
    fresh_policy = _policy()
    trained_eval = _evaluate(
        trained_policy,
        seed + 20_000,
        candidate_depths=TRANSFER_DEPTHS,
    )
    fresh_eval = _evaluate(
        fresh_policy,
        seed + 21_000,
        candidate_depths=TRANSFER_DEPTHS,
    )
    shuffled_eval = _evaluate(
        shuffled_policy,
        seed + 22_000,
        candidate_depths=TRANSFER_DEPTHS,
    )
    corrupted_eval = _evaluate(
        trained_policy,
        seed + 23_000,
        candidate_depths=TRANSFER_DEPTHS,
        corrupt_features=True,
    )
    persisted = _policy()
    persisted.load_state_dict(trained_policy.state_dict())
    gates = {
        "trained_beats_fresh": (
            trained_eval["accuracy"] >= fresh_eval["accuracy"] + CAUSAL_MARGIN
        ),
        "trained_beats_shuffled": (
            trained_eval["accuracy"] >= shuffled_eval["accuracy"] + CAUSAL_MARGIN
        ),
        "feature_causal": trained_eval["accuracy"] > corrupted_eval["accuracy"] + 0.20,
        "order_permutation": trained_eval["accuracy"] >= 0.80,
        "policy_persistence_exact": _digest(persisted) == _digest(trained_policy),
        "stable_transfer_threshold": training["stable_update"] is not None,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.learned-recipe-eviction.v3",
        "claim_boundary": (
            f"bounded context-conditioned learned victim choice over "
            f"{CANDIDATE_COUNT} recursive recipe files from "
            "permutation-safe structural telemetry and scalar compaction "
            "utility; not universal eviction economics or general continual "
            "learning"
        ),
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "train_candidate_depths": list(TRAIN_DEPTHS),
            "transfer_candidate_depths": list(TRANSFER_DEPTHS),
            "capacity_pressure_regimes": CANDIDATE_COUNT,
            "active_root_rule": "pressure_regime_selects_required_depth_rank",
            "credit_mode": credit_mode,
            "stable_accuracy_threshold": STABLE_ACCURACY_THRESHOLD,
            "causal_margin": CAUSAL_MARGIN,
            "probe_interval": PROBE_INTERVAL,
            "probe_episodes": PROBE_EPISODES,
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
            "optimizer_updates": int(training["optimizer_updates"])
            + int(shuffled_training["optimizer_updates"]),
            "primary_policy_optimizer_updates": TRAIN_EPISODES,
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
    global TRAIN_EPISODES, EVAL_EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--candidate-count", type=int, choices=tuple(DEPTH_PROFILES), default=2)
    parser.add_argument("--credit-mode", choices=CREDIT_MODES, default="sampled")
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.train_episodes < 1 or args.eval_episodes < 1:
        parser.error("episode counts must be positive")
    TRAIN_EPISODES = args.train_episodes
    EVAL_EPISODES = args.eval_episodes
    report = run(
        args.seed,
        args.report_out,
        candidate_count=args.candidate_count,
        credit_mode=args.credit_mode,
    )
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
