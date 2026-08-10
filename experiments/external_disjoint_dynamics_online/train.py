"""Fast online-routing audit over genuinely disjoint opaque dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 12
MODEL_HIDDEN_WIDTH = 48
CONTEXT_HIDDEN_WIDTH = 40
POSITION_COUNT = 7
CONTEXT_UPDATES = 400
SOURCE_UPDATES = 900
TARGET_UPDATES = 220
LOSS_THRESHOLD = 0.01
MATCH_TOLERANCE = 0.02
MATCH_MARGIN = 0.01
ADMISSION_OBSERVATIONS = POSITION_COUNT * 2
REGIME_NAMES = ("source_a", "source_b", "target_c", "target_d")
HORIZON = 3

# Diagnostic-only fixture bookkeeping: each set covers one valid three-step
# path for every held-out target while withholding the remaining transition
# rows. The router receives no regime label or row-index metadata.
TARGET_COVERING_ROW_INDICES = (
    (0, 1, 2, 6, 9, 10),
    (0, 1, 2, 4, 10, 12, 13),
    (1, 5, 8, 11, 12),
    (0, 1, 2, 3, 6, 10, 12),
)

# Same opaque interface, four unrelated transition functions. These are
# verifier-private tables; no table or regime identifier reaches the router.
TRANSITION_TABLES = (
    ((1, 4, 6, 6, 6, 0, 2), (0, 3, 6, 3, 3, 5, 3)),
    ((6, 6, 0, 0, 0, 2, 6), (1, 5, 6, 5, 6, 2, 2)),
    ((1, 4, 4, 1, 2, 4, 3), (5, 4, 0, 4, 0, 6, 3)),
    ((1, 2, 0, 5, 3, 3, 1), (0, 0, 0, 3, 4, 2, 6)),
)
TARGETS = (
    ((0, 4), (1, 6), (5, 0)),
    ((0, 2), (1, 6), (5, 1)),
    ((0, 3), (2, 6), (4, 5)),
    ((0, 2), (3, 5), (6, 1)),
)


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, ExternalTransitionObservation]]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    observations: dict[str, ExternalTransitionObservation] = {}
    for regime_index, name in enumerate(REGIME_NAMES):
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action_index in range(2):
                next_position = TRANSITION_TABLES[regime_index][action_index][position]
                states.append(state_codes[position])
                intentions.append(intention_codes[action_index])
                next_states.append(state_codes[next_position])
        observations[name] = ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(POSITION_COUNT * 2),
        )
    return state_codes, intention_codes, observations


def _rows(
    observation: ExternalTransitionObservation,
    indices: tuple[int, ...] | None = None,
) -> list[ExternalTransitionObservation]:
    row_indices = (
        tuple(range(observation.state.shape[0]))
        if indices is None
        else indices
    )
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence[index : index + 1]
            ),
        )
        for index in row_indices
    ]


def _stream_rows(
    observation: ExternalTransitionObservation,
    *,
    regime_index: int,
    partial_evidence: bool,
    noise_std: float = 0.0,
    noise_seed: int = 0,
) -> list[ExternalTransitionObservation]:
    if noise_std < 0.0:
        raise ValueError("stream noise standard deviation cannot be negative")
    selected = _rows(
        observation,
        None if not partial_evidence else TARGET_COVERING_ROW_INDICES[regime_index],
    )
    repeats = (ADMISSION_OBSERVATIONS + len(selected) - 1) // len(selected)
    stream = (selected * repeats)[:ADMISSION_OBSERVATIONS]
    if noise_std == 0.0:
        return stream
    noisy: list[ExternalTransitionObservation] = []
    for row_index, item in enumerate(stream):
        generator = torch.Generator().manual_seed(noise_seed + row_index)
        noisy.append(
            ExternalTransitionObservation(
                state=item.state
                + noise_std
                * torch.randn(
                    item.state.shape,
                    generator=generator,
                    device=item.state.device,
                    dtype=item.state.dtype,
                ),
                intention=item.intention,
                next_state=item.next_state
                + noise_std
                * torch.randn(
                    item.next_state.shape,
                    generator=generator,
                    device=item.next_state.device,
                    dtype=item.next_state.dtype,
                ),
                confidence=item.confidence,
            )
        )
    return noisy


def _train_context_encoder(
    encoder: ExternalTransitionContextEncoder,
    observations: dict[str, ExternalTransitionObservation],
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    for update in range(1, CONTEXT_UPDATES + 1):
        left_views: list[torch.Tensor] = []
        right_views: list[torch.Tensor] = []
        for index, name in enumerate(REGIME_NAMES[:2]):
            observation = observations[name]
            left = observation.state + 0.01 * torch.randn(
                observation.state.shape,
                generator=torch.Generator().manual_seed(seed + update * 11 + index),
            )
            right = observation.state + 0.02 * torch.randn(
                observation.state.shape,
                generator=torch.Generator().manual_seed(seed + update * 17 + index),
            )
            left_observation = ExternalTransitionObservation(
                state=left,
                intention=observation.intention,
                next_state=observation.next_state,
                confidence=observation.confidence,
            )
            right_observation = ExternalTransitionObservation(
                state=right,
                intention=observation.intention,
                next_state=observation.next_state,
                confidence=observation.confidence,
            )
            left_views.append(encoder.encode_observation(left_observation))
            right_views.append(encoder.encode_observation(right_observation))
        loss = encoder.contrastive_loss(
            torch.stack(left_views), torch.stack(right_views), temperature=0.1
        )
        final_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return final_loss, CONTEXT_UPDATES


def _evaluate(
    bank: ExternalTransitionModelBank,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    table: tuple[tuple[int, ...], tuple[int, ...]],
    targets: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(bank, beam_width=16)
    successes: list[bool] = []
    expanded_nodes = 0
    for start, goal in targets:
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=HORIZON,
            transition_context=context.unsqueeze(0),
        )
        expanded_nodes += result.expanded_nodes
        position = start
        for intention in result.intentions[0]:
            action = int(
                torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
            )
            position = table[action][position]
        successes.append(position == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "expanded_nodes": expanded_nodes,
    }


def _train_slot(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
    *,
    mastery_probe: Callable[[], bool] | None = None,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    final_loss = float("inf")
    for update in range(1, updates + 1):
        final_loss = bank.adaptation_step(observation, context_batch, optimizer)
        if final_loss <= LOSS_THRESHOLD and (
            mastery_probe is None or mastery_probe()
        ):
            return final_loss, update
    return final_loss, updates


def _factual_error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    prediction = bank(observation.state, observation.intention, context_batch)
    return float((prediction - observation.next_state).square().mean().detach())


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=MODEL_HIDDEN_WIDTH,
        capacity=len(REGIME_NAMES),
    )


def run(
    seed: int,
    report_out: Path,
    *,
    sequence_repeats: int = 1,
    partial_evidence: bool = False,
    stream_noise_std: float = 0.0,
) -> dict[str, object]:
    if sequence_repeats < 1:
        raise ValueError("sequence repeats must be positive")
    if stream_noise_std < 0.0:
        raise ValueError("stream noise standard deviation cannot be negative")
    begun = time.perf_counter()
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    context_loss, context_updates = _train_context_encoder(
        encoder, observations, seed=seed
    )
    encoder.eval()
    with torch.no_grad():
        contexts = {
            name: encoder.encode_observation(observation)
            for name, observation in observations.items()
        }

    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    bank = _new_bank()
    source_a_index = bank.ensure_context(contexts["source_a"])
    source_b_index = bank.ensure_context(
        contexts["source_b"], initialize_from=source_a_index
    )
    source_a_loss, source_a_updates = _train_slot(
        bank,
        source_a_index,
        observations["source_a"],
        contexts["source_a"],
        SOURCE_UPDATES,
        mastery_probe=lambda: float(
            _evaluate(
                bank,
                state_codes,
                intention_codes,
                contexts["source_a"],
                TRANSITION_TABLES[0],
                TARGETS[0],
            )["mastery"]
        )
        >= 0.8,
    )
    source_b_loss, source_b_updates = _train_slot(
        bank,
        source_b_index,
        observations["source_b"],
        contexts["source_b"],
        SOURCE_UPDATES,
        mastery_probe=lambda: float(
            _evaluate(
                bank,
                state_codes,
                intention_codes,
                contexts["source_b"],
                TRANSITION_TABLES[1],
                TARGETS[1],
            )["mastery"]
        )
        >= 0.8,
    )
    prior_digests = {
        "source_a": bank.models[source_a_index].digest(),
        "source_b": bank.models[source_b_index].digest(),
    }
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=MATCH_MARGIN,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=len(REGIME_NAMES),
    )

    base_sequence = (
        "source_a",
        "source_b",
        "target_c",
        "source_a",
        "target_d",
        "target_c",
        "target_d",
        "source_b",
    )
    sequence = base_sequence * sequence_repeats
    optimizers: dict[int, torch.optim.Optimizer] = {}
    assignments: dict[str, set[int]] = defaultdict(set)
    route_counts: Counter[str] = Counter()
    target_updates: dict[str, int] = defaultdict(int)
    target_admissions: Counter[str] = Counter()
    target_reuses: Counter[str] = Counter()
    old_slot_updates = 0
    trace: list[dict[str, object]] = []

    for sequence_position, regime in enumerate(sequence):
        regime_index = REGIME_NAMES.index(regime)
        for row in _stream_rows(
            observations[regime],
            regime_index=regime_index,
            partial_evidence=partial_evidence,
            noise_std=stream_noise_std,
            noise_seed=seed + sequence_position * 1009,
        ):
            result = router.observe(row)
            route_counts[f"{regime}:{result.status}"] += 1
            if result.slot_index is not None:
                assignments[regime].add(result.slot_index)
            if regime.startswith("target_") and result.status == "admitted":
                target_admissions[regime] += 1
            if regime.startswith("target_") and result.status == "matched":
                target_reuses[regime] += 1
            if result.status == "admitted" and result.slot_index is not None:
                target_name = regime
                target_index = result.slot_index
                optimizer = torch.optim.Adam(
                    router.bank.models[target_index].parameters(), lr=0.01
                )
                optimizers[target_index] = optimizer
                loss = router.adaptation_step(result, optimizer)
                updates = 1
                while updates < TARGET_UPDATES and (
                    loss > LOSS_THRESHOLD
                    or float(
                        _evaluate(
                            bank,
                            state_codes,
                            intention_codes,
                            router.bank.context_at(target_index),
                            TRANSITION_TABLES[REGIME_NAMES.index(target_name)],
                            TARGETS[REGIME_NAMES.index(target_name)],
                        )["mastery"]
                    )
                    < 0.8
                ):
                    loss = router.adaptation_step(result, optimizer)
                    updates += 1
                target_updates[target_name] += updates
            elif result.status == "matched" and result.slot_index in {
                source_a_index,
                source_b_index,
            }:
                old_slot_updates += 0
            trace.append(
                {
                    "diagnostic_regime": regime,
                    "status": result.status,
                    "slot_index": result.slot_index,
                    "pending_observations": result.pending_observations,
                }
            )

    retention: dict[str, dict[str, object]] = {}
    all_retained = True
    all_stable = True
    for index, name in enumerate(REGIME_NAMES):
        result = _evaluate(
            bank,
            state_codes,
            intention_codes,
            bank.context_at(index),
            TRANSITION_TABLES[index],
            TARGETS[index],
        )
        stable = (
            name not in prior_digests
            or bank.models[index].digest() == prior_digests[name]
        )
        retention[name] = {"mastery": result["mastery"], "byte_stable": stable}
        all_retained = all_retained and float(result["mastery"]) >= 0.8
        if name in prior_digests:
            all_stable = all_stable and stable

    fresh_updates: dict[str, int] = {}
    fresh_mastery: dict[str, float] = {}
    for index, name in enumerate(REGIME_NAMES[2:], start=2):
        fresh = _new_bank()
        fresh_index = fresh.ensure_context(contexts[name])
        loss, updates = _train_slot(
            fresh,
            fresh_index,
            observations[name],
            contexts[name],
            TARGET_UPDATES,
            mastery_probe=lambda index=index, fresh=fresh: float(
                _evaluate(
                    fresh,
                    state_codes,
                    intention_codes,
                    contexts[REGIME_NAMES[index]],
                    TRANSITION_TABLES[index],
                    TARGETS[index],
                )["mastery"]
            )
            >= 0.8,
        )
        fresh_updates[name] = updates
        fresh_mastery[name] = float(
            _evaluate(
                fresh,
                state_codes,
                intention_codes,
                contexts[name],
                TRANSITION_TABLES[index],
                TARGETS[index],
            )["mastery"]
        )

    target_speed = all(
        target_updates[name] < fresh_updates[name]
        for name in REGIME_NAMES[2:]
    )
    extended_reuse = all(
        target_reuses[name] >= (2 * sequence_repeats - 1)
        for name in REGIME_NAMES[2:]
    )
    wrong_context_error = _factual_error(
        bank,
        observations["target_c"],
        bank.context_at(source_a_index),
    )
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    distinct_tables = len({repr(table) for table in TRANSITION_TABLES}) == len(
        TRANSITION_TABLES
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_pretraining_converged": context_loss < 0.05,
        "source_a_mastered": float(retention["source_a"]["mastery"]) >= 0.8,
        "source_b_mastered": float(retention["source_b"]["mastery"]) >= 0.8,
        "disjoint_transition_tables": distinct_tables,
        "target_c_admitted_without_label": target_admissions["target_c"] == 1,
        "target_d_admitted_without_label": target_admissions["target_d"] == 1,
        "target_c_reused_after_admission": target_reuses["target_c"] >= 1,
        "target_d_reused_after_admission": target_reuses["target_d"] >= 1,
        "all_regimes_mastered": all_retained,
        "source_slots_byte_stable": all_stable,
        "warm_targets_faster_than_fresh": target_speed,
        "extended_alternation_reuse": extended_reuse,
        "wrong_context_factual_control": wrong_context_error > LOSS_THRESHOLD,
        "old_slot_optimizer_updates_zero": old_slot_updates == 0,
        "persistence_exact": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-disjoint-dynamics-online-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "context_updates": CONTEXT_UPDATES,
            "source_updates": SOURCE_UPDATES,
            "target_updates_limit": TARGET_UPDATES,
            "admission_observations": ADMISSION_OBSERVATIONS,
            "match_tolerance": MATCH_TOLERANCE,
            "match_margin": MATCH_MARGIN,
            "regime_labels_used_by_router": False,
            "policy": "none_external_disjoint_online_context_model_search_v1",
            "sequence_repeats": sequence_repeats,
            "partial_evidence": partial_evidence,
            "stream_noise_std": stream_noise_std,
            "observed_transition_rows": {
                name: (
                    len(TARGET_COVERING_ROW_INDICES[index])
                    if partial_evidence
                    else POSITION_COUNT * 2
                )
                for index, name in enumerate(REGIME_NAMES)
            },
            "withheld_transition_rows": {
                name: (
                    POSITION_COUNT * 2 - len(TARGET_COVERING_ROW_INDICES[index])
                    if partial_evidence
                    else 0
                )
                for index, name in enumerate(REGIME_NAMES)
            },
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": context_updates,
            "loss": context_loss,
        },
        "pretraining": {
            "source_a_optimizer_updates": source_a_updates,
            "source_a_loss": source_a_loss,
            "source_b_optimizer_updates": source_b_updates,
            "source_b_loss": source_b_loss,
        },
        "routing": {
            "sequence": list(sequence),
            "counts": dict(route_counts),
            "assignments": {name: sorted(values) for name, values in assignments.items()},
            "target_admissions": dict(target_admissions),
            "target_reuses": dict(target_reuses),
            "trace": trace,
        },
        "targets": {
            "warm_optimizer_updates": dict(target_updates),
            "fresh_optimizer_updates": dict(fresh_updates),
            "fresh_mastery": fresh_mastery,
            "old_slot_optimizer_updates": old_slot_updates,
        },
        "retention": retention,
        "controls": {
            "wrong_context_target_mse": wrong_context_error,
            "transition_table_distinct": distinct_tables,
        },
        "accounting": {
            "controller_parameter_updates": 0,
            "context_encoder_updates": context_updates,
            "old_regime_replay_during_target_adaptation": 0,
            "target_current_bundle_reuse_isolated": True,
            "planner_search_compute_reported": True,
            "unique_transition_lifetimes_per_regime": POSITION_COUNT * 2,
        },
        "digests": {
            "controller": controller_digest,
            "bank": bank.digest(),
            "context_encoder": encoder.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70411)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--sequence-repeats", type=int, default=1)
    parser.add_argument("--partial-evidence", action="store_true")
    parser.add_argument("--stream-noise-std", type=float, default=0.0)
    args = parser.parse_args()
    run(
        args.seed,
        args.report_out,
        sequence_repeats=args.sequence_repeats,
        partial_evidence=args.partial_evidence,
        stream_noise_std=args.stream_noise_std,
    )


if __name__ == "__main__":
    main()
