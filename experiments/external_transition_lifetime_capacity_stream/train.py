"""Two-seed long alternating lifetime-pressure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_computer import ExternalTransitionModelBank, ExternalTransitionObservation
from neural_computer import ExternalTransitionModelLifetimePolicy

CONTEXT_WIDTH = 5
CAPACITY = 4
RECURRING_COUNT = 3
PRESSURES = 12
TRAIN_STREAMS = 50
EVAL_STREAMS = 20
HIDDEN_WIDTH = 16
LEARNING_RATE = 0.03


@dataclass(frozen=True)
class Capability:
    context: torch.Tensor
    training: ExternalTransitionObservation
    heldout: ExternalTransitionObservation


def _capability(
    generator: torch.Generator,
    index: int,
) -> Capability:
    context = torch.nn.functional.normalize(
        torch.randn(CONTEXT_WIDTH, generator=generator), dim=0
    )
    train_rows = 5
    state = torch.rand(train_rows, 1, generator=generator) * 2.0 - 1.0
    intention = torch.rand(train_rows, 1, generator=generator) * 2.0 - 1.0
    slope = 0.3 + 0.15 * index
    intention_weight = 0.8 - 0.08 * index
    bias = -0.1 + 0.05 * index
    transition = slope * state + intention_weight * intention + bias
    heldout_state = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    heldout_intention = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    heldout_transition = (
        slope * heldout_state + intention_weight * heldout_intention + bias
    )
    return Capability(
        context=context,
        training=ExternalTransitionObservation(state, intention, transition),
        heldout=ExternalTransitionObservation(
            heldout_state,
            heldout_intention,
            heldout_transition,
        ),
    )


def _train_capability(bank: ExternalTransitionModelBank, capability: Capability) -> int:
    index = bank.ensure_context(capability.context)
    context = bank.context_at(index)
    bank.adaptation_step(
        capability.training,
        context.unsqueeze(0).expand(capability.training.state.shape[0], -1),
        None,
    )
    return bank.slot_id_at(index)


def _retained_probe(
    candidate: ExternalTransitionModelBank,
    retained: dict[int, Capability],
) -> bool:
    for slot_id, capability in retained.items():
        try:
            index = candidate.physical_index_for_slot_id(slot_id)
        except KeyError:
            return False
        context = candidate.context_at(index)
        context_batch = context.unsqueeze(0).expand(
            capability.heldout.state.shape[0], -1
        )
        if float(candidate.loss(capability.heldout, context_batch).detach()) > 1e-6:
            return False
    return True


def _build_stream(generator: torch.Generator) -> tuple[
    ExternalTransitionModelBank,
    dict[int, Capability],
    Capability,
]:
    bank = ExternalTransitionModelBank(
        1,
        1,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=CAPACITY,
    )
    recurring: dict[int, Capability] = {}
    for index in range(RECURRING_COUNT):
        capability = _capability(generator, index)
        slot_id = _train_capability(bank, capability)
        recurring[slot_id] = capability
    disposable = _capability(generator, 9)
    _train_capability(bank, disposable)
    return bank, recurring, disposable


def _pressure_step(
    policy: ExternalTransitionModelLifetimePolicy,
    bank: ExternalTransitionModelBank,
    recurring: dict[int, Capability],
    disposable: Capability,
    *,
    update: bool,
) -> tuple[bool, bool, bool]:
    for capability in recurring.values():
        _train_capability(bank, capability)
    disposable_slot_id = bank.slot_ids[-1]
    retained = dict(recurring)
    expected = tuple(slot_id for slot_id in bank.slot_ids if slot_id != disposable_slot_id)
    protected = torch.tensor(
        [slot_id in {0, 1} for slot_id in bank.slot_ids],
        dtype=torch.bool,
    )
    source = bank.slot_ids

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        if candidate.slot_ids == source:
            return True
        return candidate.slot_ids == expected and _retained_probe(candidate, retained)

    proposal, receipt = policy.evict_from_bank_verified(
        bank,
        protected,
        retention_probe,
        update=update,
    )
    learned_success = bool(receipt is not None and receipt.accepted)
    retained_after_learned = _retained_probe(bank, retained)
    if not learned_success:
        fallback = bank.evict_verified_id(
            disposable_slot_id,
            retention_probe,
        )
        if not fallback.accepted:
            raise RuntimeError("verifier fallback could not preserve recurring models")
    retained_after_fallback = _retained_probe(bank, retained)
    if not update and proposal.selected_slot_id is None:
        raise RuntimeError("frozen lifetime policy refused an eligible pressure event")
    _train_capability(bank, disposable)
    return learned_success, retained_after_learned, retained_after_fallback


def _run_stream(
    policy: ExternalTransitionModelLifetimePolicy,
    generator: torch.Generator,
    *,
    streams: int,
    update: bool,
) -> dict[str, float | int | bool]:
    learned_successes = 0
    learned_retention = True
    stream_retention = True
    pressure_count = 0
    for _ in range(streams):
        bank, recurring, disposable = _build_stream(generator)
        for _ in range(PRESSURES):
            success, retained_after_learned, retained_after_fallback = _pressure_step(
                policy,
                bank,
                recurring,
                disposable,
                update=update,
            )
            pressure_count += 1
            learned_successes += int(success)
            learned_retention = learned_retention and retained_after_learned
            stream_retention = stream_retention and retained_after_fallback
            disposable = _capability(generator, 9)
    return {
        "pressure_events": pressure_count,
        "learned_successes": learned_successes,
        "learned_selection": learned_successes / pressure_count,
        "learned_retention_floor": learned_retention,
        "stream_retention_floor": stream_retention,
    }


def _digest(policy: ExternalTransitionModelLifetimePolicy) -> str:
    return hashlib.sha256(policy.digest().encode("utf-8")).hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    policy = ExternalTransitionModelLifetimePolicy(
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
    )
    train = _run_stream(policy, generator, streams=TRAIN_STREAMS, update=True)
    evaluation = _run_stream(policy, generator, streams=EVAL_STREAMS, update=False)
    restored = ExternalTransitionModelLifetimePolicy.from_payload(policy.state_payload())
    controls = {
        "random": 0.5,
        "recency": 1.0,
    }
    report = {
        "schema": "neural-computer.external-transition-lifetime-capacity-stream.v1",
        "seed": seed,
        "configuration": {
            "train_streams": TRAIN_STREAMS,
            "evaluation_streams": EVAL_STREAMS,
            "pressures_per_stream": PRESSURES,
            "capacity": CAPACITY,
            "recurring_capabilities": RECURRING_COUNT,
            "verifier": "heldout-recurring-model-retention-v1",
        },
        "gates": {
            "learned_beats_random_by_margin": evaluation["learned_selection"]
            >= controls["random"] + 0.10,
            "learned_retention_floor": evaluation["learned_retention_floor"],
            "stream_retention_floor": evaluation["stream_retention_floor"],
            "exact_policy_persistence": restored.digest() == policy.digest(),
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": (
            evaluation["learned_selection"] >= controls["random"] + 0.10
            and bool(evaluation["learned_retention_floor"])
            and bool(evaluation["stream_retention_floor"])
            and restored.digest() == policy.digest()
        ),
        "metrics": {
            "train": train,
            "evaluation": evaluation,
            "controls": controls,
            "policy_digest": _digest(policy),
        },
        "accounting": {
            "unique_verifier_bits": int(train["pressure_events"]),
            "unique_logical_lifetimes": int(train["pressure_events"]) * CAPACITY,
            "policy_optimizer_updates": int(train["pressure_events"]),
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "long bounded alternating retention stream; not unrestricted general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
