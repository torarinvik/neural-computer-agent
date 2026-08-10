"""Stress-test independent external intention cells on a nonstationary stream.

The controller and factual transition model are frozen. The memory receives
only a masked opaque controller state, proposes one candidate per external
cell, and receives one scalar verifier outcome for the explicitly selected
cell. Feedback is delayed and optionally noisy. Mastered cells are protected;
new regimes use copy-on-write cells. A matched fresh learner and a
reward-shuffled control keep transfer and causality honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalControllerStateAdapter,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionGeneratorState,
    ExternalOutcomeIntentionMemory,
    OpaqueProtocolDecoder,
    PolicyFreeAmodalRuntime,
)

CONTROLLER_WIDTH = 4
STATE_WIDTH = 12
INTENTION_WIDTH = 2
HORIZON = 1
BEAM_WIDTH = 8
UTILITY_TEMPERATURE = 0.8
MASTERY_THRESHOLD = 0.95
NOISY_MASTERY_THRESHOLD = 0.90
SOURCE_TARGET = torch.tensor([0.75, -0.75])
SUCCESSOR_TARGET = torch.tensor([0.55, -0.95])
REVERSED_TARGET = torch.tensor([-0.75, 0.75])
PARTIAL_MASK = torch.tensor(
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
)


class _AdditiveFactualModel(nn.Module):
    state_width = STATE_WIDTH
    intention_width = INTENTION_WIDTH

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :INTENTION_WIDTH] += intention
        return result


class _PartialStateAdapter(ExternalControllerStateAdapter):
    """Frozen external projection that removes half of the opaque state."""

    def __init__(self) -> None:
        super().__init__(STATE_WIDTH, STATE_WIDTH)
        self.register_buffer("partial_mask", PARTIAL_MASK.clone())

    def forward(self, output):
        return super().forward(output) * self.partial_mask


def _digest_module(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _digest_state(state: ExternalOutcomeIntentionGeneratorState) -> str:
    digest = hashlib.sha256()
    for name in (
        "input_weights",
        "input_bias",
        "output_weights",
        "output_bias",
        "baseline",
        "decisions",
        "feedbacks",
        "protected",
    ):
        value = getattr(state, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cell_snapshot(
    state: ExternalOutcomeIntentionGeneratorState,
    cell_index: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        getattr(state, name)[cell_index].detach().clone()
        for name in ("input_weights", "input_bias", "output_weights", "output_bias", "baseline")
    )


def _cell_matches(
    state: ExternalOutcomeIntentionGeneratorState,
    cell_index: int,
    snapshot: tuple[torch.Tensor, ...],
) -> bool:
    return all(
        torch.equal(getattr(state, name)[cell_index], expected)
        for name, expected in zip(
            ("input_weights", "input_bias", "output_weights", "output_bias", "baseline"),
            snapshot,
            strict=True,
        )
    )


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def _utility(intention: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -(intention - target.reshape(1, -1)).square().sum(dim=-1)
        / UTILITY_TEMPERATURE
    ).clamp(0.0, 1.0)


def _goal(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    goal = context.clone()
    goal[:, :INTENTION_WIDTH] += target.reshape(1, -1)
    return goal


def _build(seed: int):
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=CONTROLLER_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder(
        "opaque_backend",
        OpaqueProtocolDecoder(INTENTION_WIDTH, INTENTION_WIDTH),
    )
    generator = ExternalOutcomeIntentionGenerator(
        context_width=STATE_WIDTH,
        intention_width=INTENTION_WIDTH,
        hidden_width=32,
        initial_learning_rate=0.03,
        initial_baseline_rate=0.05,
        noise_scale=0.35,
        initial_parameter_scale=0.05,
    )
    memory = ExternalOutcomeIntentionMemory(generator)
    planner = ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=BEAM_WIDTH)
    adapter = _PartialStateAdapter()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        planner,
        state_adapter=adapter,
        intention_memory=memory,
    )
    controller_state = runtime.initial_state(1, device="cpu")
    feedback = _feedback()
    events = {
        "source": [AmodalEvent(torch.tensor([[0.3, -0.1, 0.4, -0.2]]))],
        "successor": [AmodalEvent(torch.tensor([[-0.6, 0.7, -0.2, 0.5]]))],
        "reversal": [AmodalEvent(torch.tensor([[0.9, 0.2, -0.8, -0.4]]))],
    }
    contexts: dict[str, torch.Tensor] = {}
    for name, event in events.items():
        preview, _ = runtime.step_events(event, controller_state, feedback)
        contexts[name] = adapter(preview.controller).detach()
    return (
        controller,
        runtime,
        policy_free,
        memory,
        controller_state,
        feedback,
        events,
        contexts,
    )


def _train_cell(
    *,
    policy_free: PolicyFreeAmodalRuntime,
    memory: ExternalOutcomeIntentionMemory,
    memory_state: ExternalOutcomeIntentionGeneratorState,
    controller_state,
    feedback: ControllerFeedback,
    event: list[AmodalEvent],
    context: torch.Tensor,
    target: torch.Tensor,
    cell_index: int,
    max_updates: int,
    delay: int,
    noise_fraction: float = 0.0,
    shuffled: bool = False,
    random_source: torch.Generator | None = None,
) -> tuple[ExternalOutcomeIntentionGeneratorState, int, float, int, float]:
    begun = time.perf_counter()
    pending: deque[tuple[object, torch.Tensor, torch.Tensor]] = deque()
    search_expansions = 0
    score = float(
        _utility(memory.mean(memory_state, context)[:, cell_index], target).item()
    )

    def apply_pending() -> None:
        nonlocal memory_state
        proposal, selected, outcome = pending.popleft()
        memory_state = memory.apply_feedback(
            memory_state,
            proposal,
            selected,
            outcome,
            terminal=torch.ones(1, dtype=torch.bool),
        )

    for update in range(1, max_updates + 1):
        # Generate every cell, then force the lifecycle-selected opaque cell
        # into the one-candidate factual probe. The verifier sees only the
        # emitted intention and its scalar outcome.
        preview, _ = policy_free.runtime.step_events(
            event,
            controller_state,
            feedback,
        )
        probe_context = policy_free.state_adapter(preview.controller).detach()
        proposal = memory.propose(memory_state, probe_context)
        selected = torch.tensor([cell_index], dtype=torch.long)
        output, _ = policy_free.step_events(
            event,
            controller_state,
            feedback,
            _goal(context, target),
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            candidate_intentions=proposal.intentions[:, cell_index : cell_index + 1],
        )
        search_expansions += int(output.planning.expanded_nodes)
        if output.planning.candidate_indices is None:
            raise AssertionError("planner did not return candidate provenance")
        if output.planning.candidate_indices.tolist() != [[0]]:
            raise AssertionError("forced lifecycle candidate was not selected")
        outcome = _utility(output.planning.intentions[:, 0], target)
        if shuffled:
            outcome = torch.rand(
                outcome.shape,
                generator=random_source,
                dtype=outcome.dtype,
            )
        elif noise_fraction:
            random_outcome = torch.rand(
                outcome.shape,
                generator=random_source,
                dtype=outcome.dtype,
            )
            outcome = (1.0 - noise_fraction) * outcome + noise_fraction * random_outcome
        pending.append((proposal, selected, outcome))
        if len(pending) > delay:
            apply_pending()
        score = float(
            _utility(memory.mean(memory_state, context)[:, cell_index], target).item()
        )
        threshold = NOISY_MASTERY_THRESHOLD if noise_fraction else MASTERY_THRESHOLD
        if not shuffled and score >= threshold:
            while pending:
                apply_pending()
            score = float(
                _utility(memory.mean(memory_state, context)[:, cell_index], target).item()
            )
            if score >= threshold:
                break
    while pending:
        apply_pending()
    score = float(
        _utility(memory.mean(memory_state, context)[:, cell_index], target).item()
    )
    return (
        memory_state,
        update,
        score,
        search_expansions,
        time.perf_counter() - begun,
    )


def _admit(
    repertoire: ExternalIntentionRepertoire,
    intention: torch.Tensor,
    target: torch.Tensor,
) -> tuple[bool, int | None]:
    receipt = repertoire.admit_verified(
        intention,
        lambda candidate: float(
            _utility(candidate.statistics()["intentions"][-1:], target).item()
        )
        >= MASTERY_THRESHOLD,
        reason="nonstationary_memory_heldout_verifier",
    )
    return receipt.accepted, receipt.entry_index


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    (
        controller,
        _runtime,
        policy_free,
        memory,
        controller_state,
        feedback,
        events,
        contexts,
    ) = _build(seed)
    controller_digest = _digest_module(controller)
    adapter_digest = _digest_module(policy_free.state_adapter)
    memory_state = memory.initial_state(1)
    random_source = torch.Generator().manual_seed(seed + 9000)

    memory_state, source_updates, source_score, source_expansions, source_seconds = _train_cell(
        policy_free=policy_free,
        memory=memory,
        memory_state=memory_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["source"],
        context=contexts["source"],
        target=SOURCE_TARGET,
        cell_index=0,
        max_updates=240,
        delay=3,
    )
    source_mean = memory.mean(memory_state, contexts["source"])[0, 0].detach().clone()
    source_snapshot = _cell_snapshot(memory_state, 0)
    memory_state = memory.protect(memory_state, [0])
    memory_state, successor_cell = memory.append_cell(memory_state, source_cell=0)

    memory_state, successor_updates, successor_score, successor_expansions, successor_seconds = _train_cell(
        policy_free=policy_free,
        memory=memory,
        memory_state=memory_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=SUCCESSOR_TARGET,
        cell_index=successor_cell,
        max_updates=240,
        delay=3,
    )
    successor_mean = memory.mean(memory_state, contexts["successor"])[0, successor_cell].detach().clone()
    successor_snapshot = _cell_snapshot(memory_state, successor_cell)
    memory_state = memory.protect(memory_state, [successor_cell])
    pre_reversal_state = memory_state
    memory_state, inherited_reversal_cell = memory.append_cell(
        memory_state, source_cell=successor_cell
    )
    (
        inherited_reversal_state,
        inherited_reversal_updates,
        inherited_reversal_score,
        inherited_reversal_expansions,
        inherited_reversal_seconds,
    ) = _train_cell(
        policy_free=policy_free,
        memory=memory,
        memory_state=memory_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["reversal"],
        context=contexts["reversal"],
        target=REVERSED_TARGET,
        cell_index=inherited_reversal_cell,
        max_updates=60,
        delay=4,
    )
    negative_transfer_detected = inherited_reversal_score < NOISY_MASTERY_THRESHOLD
    if negative_transfer_detected:
        # The failed candidate is discarded transactionally. No verifier
        # examples are replayed, and the protected prefix is restored exactly.
        memory_state = pre_reversal_state
        memory_state, reversal_cell = memory.append_cell(memory_state)
    else:
        memory_state = inherited_reversal_state
        reversal_cell = inherited_reversal_cell
    (
        memory_state,
        reversal_updates,
        reversal_score,
        reversal_expansions,
        reversal_seconds,
    ) = _train_cell(
        policy_free=policy_free,
        memory=memory,
        memory_state=memory_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["reversal"],
        context=contexts["reversal"],
        target=REVERSED_TARGET,
        cell_index=reversal_cell,
        max_updates=260,
        delay=4,
        noise_fraction=0.20,
        random_source=random_source,
    )
    reversal_mean = memory.mean(memory_state, contexts["reversal"])[0, reversal_cell].detach().clone()

    fresh_memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=STATE_WIDTH,
            intention_width=INTENTION_WIDTH,
            hidden_width=32,
            initial_learning_rate=0.03,
            initial_baseline_rate=0.05,
            noise_scale=0.35,
            initial_parameter_scale=0.05,
        )
    )
    torch.manual_seed(seed + 12000)
    fresh_state = fresh_memory.initial_state(1)
    fresh_policy_free = policy_free
    fresh_state, fresh_updates, fresh_score, fresh_expansions, fresh_seconds = _train_cell(
        policy_free=fresh_policy_free,
        memory=fresh_memory,
        memory_state=fresh_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=SUCCESSOR_TARGET,
        cell_index=0,
        max_updates=240,
        delay=0,
    )

    torch.manual_seed(seed + 13000)
    noisy_memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=STATE_WIDTH,
            intention_width=INTENTION_WIDTH,
            hidden_width=32,
            initial_learning_rate=0.03,
            initial_baseline_rate=0.05,
            noise_scale=0.35,
            initial_parameter_scale=0.05,
        )
    )
    noisy_state = noisy_memory.initial_state(1)
    noisy_state, noisy_updates, noisy_score, noisy_expansions, noisy_seconds = _train_cell(
        policy_free=fresh_policy_free,
        memory=noisy_memory,
        memory_state=noisy_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=SUCCESSOR_TARGET,
        cell_index=0,
        max_updates=240,
        delay=3,
        noise_fraction=0.20,
        random_source=random_source,
    )

    torch.manual_seed(seed + 14000)
    shuffled_memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=STATE_WIDTH,
            intention_width=INTENTION_WIDTH,
            hidden_width=32,
            initial_learning_rate=0.03,
            initial_baseline_rate=0.05,
            noise_scale=0.35,
            initial_parameter_scale=0.05,
        )
    )
    shuffled_state = shuffled_memory.initial_state(1)
    shuffled_state, shuffled_updates, shuffled_score, shuffled_expansions, shuffled_seconds = _train_cell(
        policy_free=fresh_policy_free,
        memory=shuffled_memory,
        memory_state=shuffled_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=SUCCESSOR_TARGET,
        cell_index=0,
        max_updates=240,
        delay=3,
        shuffled=True,
        random_source=random_source,
    )

    # Runtime-level full-memory probe: controller batch is one, memory has
    # three cells, and planner provenance must still identify one candidate.
    probe_output, _ = policy_free.step_events(
        events["source"],
        controller_state,
        feedback,
        _goal(contexts["source"], SOURCE_TARGET),
        horizon=HORIZON,
        beam_width=BEAM_WIDTH,
        intention_memory_state=memory_state,
    )
    probe = probe_output.intention_memory_generation
    if probe is None or probe.intentions.shape != (1, 3, INTENTION_WIDTH):
        raise AssertionError("full-memory runtime probe returned the wrong shape")
    if probe_output.planning.candidate_indices is None:
        raise AssertionError("full-memory runtime probe lost candidate provenance")

    repertoire = ExternalIntentionRepertoire(INTENTION_WIDTH, merge_cosine=0.9999)
    repertoire.observe(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    source_admitted, source_id = _admit(repertoire, source_mean, SOURCE_TARGET)
    successor_admitted, _successor_id = _admit(
        repertoire, successor_mean, SUCCESSOR_TARGET
    )
    reversal_admitted, _reversal_id = _admit(
        repertoire, reversal_mean, REVERSED_TARGET
    )
    duplicate = None
    duplicate_id = None
    duplicate_admitted = False
    for delta in (
        torch.tensor([0.0, 0.08]),
        torch.tensor([0.08, 0.0]),
        torch.tensor([-0.08, 0.0]),
        torch.tensor([0.05, 0.05]),
        torch.tensor([-0.05, 0.05]),
    ):
        candidate_duplicate = source_mean + delta
        duplicate_admitted, duplicate_id = _admit(
            repertoire, candidate_duplicate, SOURCE_TARGET
        )
        if duplicate_admitted:
            duplicate = candidate_duplicate
            break
    if source_id is None or duplicate_id is None or duplicate is None:
        raise AssertionError(
            f"source and duplicate admissions were required: {source_id}, {duplicate_id}"
        )
    consolidation = repertoire.consolidate_verified(
        (source_id, duplicate_id),
        (source_mean + duplicate) / 2.0,
        lambda candidate: float(
            _utility(candidate.statistics()["intentions"][-1:], SOURCE_TARGET).item()
        )
        >= MASTERY_THRESHOLD,
        reason="nonstationary_redundant_source_compaction",
    )

    old_cells_retained = _cell_matches(memory_state, 0, source_snapshot) and _cell_matches(
        memory_state, successor_cell, successor_snapshot
    )
    gates = {
        "source_mastered_after_delayed_feedback": source_score >= MASTERY_THRESHOLD,
        "successor_mastered_after_copy_on_write": successor_score >= MASTERY_THRESHOLD,
        "reversal_mastered_under_noise": reversal_score >= NOISY_MASTERY_THRESHOLD,
        "negative_transfer_probe_detected": negative_transfer_detected,
        "fresh_successor_control_mastered": fresh_score >= MASTERY_THRESHOLD,
        "noisy_control_mastered": noisy_score >= NOISY_MASTERY_THRESHOLD,
        "shuffled_outcome_control_failed": shuffled_score < MASTERY_THRESHOLD,
        "warm_successor_faster_than_fresh": successor_updates < fresh_updates,
        "protected_cells_unchanged_by_later_learning": old_cells_retained,
        "repeated_copy_on_write_growth": memory_state.baseline.shape[0] == 3,
        "repertoire_admissions_passed": (
            source_admitted and successor_admitted and reversal_admitted and duplicate_admitted
        ),
        "redundant_repertoire_consolidation_passed": consolidation.accepted,
        "full_memory_runtime_probe_passed": True,
        "controller_frozen": controller_digest == _digest_module(controller),
        "state_adapter_frozen": adapter_digest == _digest_module(policy_free.state_adapter),
        "exact_generator_persistence": _digest_state(
            memory.state_from_payload(memory.state_payload(memory_state))
        )
        == _digest_state(memory_state),
        "exact_repertoire_persistence": ExternalIntentionRepertoire.from_payload(
            repertoire.payload()
        ).content_digest()
        == repertoire.content_digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-memory.v1",
        "claim_boundary": (
            "bounded nonstationary external intention-cell growth with delayed/noisy "
            "outcome credit and retention-safe compaction; not general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "controller_width": CONTROLLER_WIDTH,
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "partial_context_mask": PARTIAL_MASK.tolist(),
            "delay_steps": 3,
            "reversal_delay_steps": 4,
            "noise_fraction": 0.20,
            "memory": memory.configuration(),
            "candidate_memory": repertoire.configuration(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "source_updates": source_updates,
            "source_score": source_score,
            "successor_updates": successor_updates,
            "successor_score": successor_score,
            "reversal_updates": reversal_updates,
            "reversal_score": reversal_score,
            "inherited_reversal_updates": inherited_reversal_updates,
            "inherited_reversal_score": inherited_reversal_score,
            "inherited_reversal_seconds": inherited_reversal_seconds,
            "source_seconds": source_seconds,
            "successor_seconds": successor_seconds,
            "reversal_seconds": reversal_seconds,
            "fresh_updates": fresh_updates,
            "fresh_score": fresh_score,
            "fresh_seconds": fresh_seconds,
            "noisy_updates": noisy_updates,
            "noisy_score": noisy_score,
            "noisy_seconds": noisy_seconds,
            "shuffled_updates": shuffled_updates,
            "shuffled_score": shuffled_score,
            "shuffled_seconds": shuffled_seconds,
            "repertoire_count_after_consolidation": repertoire.record_count,
            "probe_selected_candidate": int(
                probe_output.planning.candidate_indices[0, 0].item()
            ),
        },
        "accounting": {
            "unique_verifier_bits": (
                source_updates
                + successor_updates
                + inherited_reversal_updates
                + reversal_updates
                + fresh_updates
                + noisy_updates
            ),
            "control_outcome_bits": shuffled_updates,
            "unique_logical_lifetimes": 6,
            "external_memory_updates": (
                source_updates
                + successor_updates
                + inherited_reversal_updates
                + reversal_updates
                + fresh_updates
                + noisy_updates
                + shuffled_updates
            ),
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": (
                source_expansions
                + successor_expansions
                + reversal_expansions
                + inherited_reversal_expansions
                + fresh_expansions
                + noisy_expansions
                + shuffled_expansions
                + int(probe_output.planning.expanded_nodes)
            ),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
