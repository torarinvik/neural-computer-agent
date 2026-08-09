"""Pressure test noise-tolerant learned evidence admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineContextAddressResolver,
    ExternalTransitionEvidenceEvaluator,
    ExternalTransitionMemory,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
HIDDEN_WIDTH = 32
POSITION_COUNT = 3
EVIDENCE_ROWS = 512
EVIDENCE_UPDATES = 500
NOISE_SCALE = 0.08


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[
    tuple[ExternalTransitionObservation, ...],
    tuple[ExternalTransitionObservation, ...],
    tuple[ExternalTransitionObservation, ...],
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    positions = (0, 1, 2)
    actions = (0, 1, 0)

    def rows(deltas: tuple[int, int]) -> tuple[ExternalTransitionObservation, ...]:
        result: list[ExternalTransitionObservation] = []
        for position, action in zip(positions, actions, strict=True):
            next_position = min(
                POSITION_COUNT - 1, max(0, position + deltas[action])
            )
            result.append(
                ExternalTransitionObservation(
                    state=state_codes[position].unsqueeze(0),
                    intention=intention_codes[action].unsqueeze(0),
                    next_state=state_codes[next_position].unsqueeze(0),
                    confidence=torch.ones(1),
                )
            )
        return tuple(result)

    stream_keys = F.normalize(
        torch.randn(3, CONTEXT_WIDTH, generator=generator), dim=-1
    )
    return rows((1, -1)), rows((1, -1)), rows((-1, 1)), stream_keys


def _train_evaluator(seed: int) -> tuple[ExternalTransitionEvidenceEvaluator, float]:
    torch.manual_seed(seed)
    prediction = F.normalize(torch.randn(EVIDENCE_ROWS, STATE_WIDTH), dim=-1)
    positive = prediction + torch.randn_like(prediction) * NOISE_SCALE
    negative = F.normalize(torch.randn_like(prediction), dim=-1)
    inputs = torch.cat((prediction, prediction), dim=0)
    observations = torch.cat((positive, negative), dim=0)
    outcomes = torch.cat(
        (torch.ones(EVIDENCE_ROWS), torch.zeros(EVIDENCE_ROWS))
    )
    hits = torch.ones(EVIDENCE_ROWS * 2)
    evaluator = ExternalTransitionEvidenceEvaluator(
        STATE_WIDTH, hidden_width=HIDDEN_WIDTH
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.02)
    final_loss = float("inf")
    for _update in range(EVIDENCE_UPDATES):
        optimizer.zero_grad()
        loss = evaluator.loss(inputs, observations, outcomes, hits)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, final_loss


def _new_memory() -> ExternalTransitionMemory:
    return ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )


def _observe(
    resolver: ExternalOnlineContextAddressResolver,
    rows: tuple[ExternalTransitionObservation, ...],
    stream_key: torch.Tensor,
    memory: ExternalTransitionMemory,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in rows:
        resolution = resolver.observe(row, stream_key, memory)
        results.append(
            {
                "status": resolution.status,
                "committed_observations": resolution.committed_observations,
                "pending_observations": resolution.pending_observations,
                "record_count": memory.record_count,
            }
        )
    return results


def _diagnostic(
    memory: ExternalTransitionMemory,
    rows: tuple[ExternalTransitionObservation, ...],
    context: torch.Tensor,
) -> dict[str, float]:
    state = torch.cat([row.state for row in rows])
    intention = torch.cat([row.intention for row in rows])
    next_state = torch.cat([row.next_state for row in rows])
    prediction, hits = memory.predict_with_hit(
        state,
        intention,
        context=context.unsqueeze(0).expand(len(rows), -1),
    )
    return {
        "hit_rate": float(hits.float().mean()),
        "next_state_mse": float((prediction - next_state).square().mean()),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    clean_rows, duplicate_rows, reversal_rows, stream_keys = _fixture(seed)
    noisy_rows = tuple(
        ExternalTransitionObservation(
            state=row.state,
            intention=row.intention,
            next_state=row.next_state + torch.randn_like(row.next_state) * NOISE_SCALE,
            confidence=row.confidence,
        )
        for row in duplicate_rows
    )
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

    memory = _new_memory()
    resolver = ExternalOnlineContextAddressResolver(
        CONTEXT_WIDTH,
        address_seed=seed + 1000,
        admission_observations=3,
        contradiction_observations=2,
    )
    clean_trace = _observe(resolver, clean_rows, stream_keys[0], memory)
    source_context = resolver.addresses()[0].clone()
    source_payload = resolver.payload()
    source_state = {
        name: value.detach().clone() for name, value in memory.store.state_dict().items()
    }
    source_record_count = memory.record_count

    evaluator, evaluator_loss = _train_evaluator(seed + 2000)
    resolver.evidence_evaluator = evaluator
    learned_trace = _observe(resolver, noisy_rows, stream_keys[1], memory)
    noisy_reuse_context = resolver.addresses()[0].clone()
    noisy_record_count = memory.record_count
    noisy_reuse = _diagnostic(memory, clean_rows, source_context)
    noisy_probe = {
        "clean_prediction_mse": noisy_reuse["next_state_mse"],
        "store_records_after_noisy_reuse": noisy_record_count,
    }

    reversal_rows = reversal_rows[:2]
    reversal_trace = _observe(resolver, reversal_rows, stream_keys[0], memory)
    reversal_context = resolver.addresses()[-1].clone()

    fixed_memory = _new_memory()
    fixed_memory.store.load_state_dict(source_state)
    fixed_resolver = ExternalOnlineContextAddressResolver.from_payload(source_payload)
    fixed_trace = _observe(fixed_resolver, noisy_rows, stream_keys[1], fixed_memory)

    wrong_context = _diagnostic(memory, reversal_rows, source_context)
    retained = _diagnostic(memory, clean_rows, source_context)
    fresh = _diagnostic(_new_memory(), clean_rows, source_context)
    restored_memory = _new_memory()
    restored_memory.store.load_state_dict(memory.store.state_dict())
    restored_resolver = ExternalOnlineContextAddressResolver.from_payload(
        resolver.payload(), evidence_evaluator=evaluator
    )
    persisted = _diagnostic(restored_memory, reversal_rows, reversal_context)
    with torch.no_grad():
        positive_probability = float(
            torch.sigmoid(
                evaluator(
                    clean_rows[0].next_state,
                    noisy_rows[0].next_state,
                    torch.ones(1),
                )
            ).item()
        )
        negative_probability = float(
            torch.sigmoid(
                evaluator(
                    clean_rows[0].next_state,
                    reversal_rows[0].next_state,
                    torch.ones(1),
                )
            ).item()
        )
    controller_unchanged = controller_digest == _digest_module(controller)
    gates = {
        "controller_unchanged": controller_unchanged,
        "evidence_evaluator_learns": evaluator_loss < 0.01,
        "learned_accepts_noisy_reuse": all(
            item["status"] == "reused" for item in learned_trace
        ),
        "learned_no_growth_on_noisy_reuse": noisy_record_count == source_record_count,
        "read_only_retention": retained["next_state_mse"] < 1e-6,
        "fixed_exact_control_rejects_noise": any(
            item["status"] == "admitted" for item in fixed_trace
        ),
        "reversal_admits_new_context": reversal_trace[-1]["status"] == "admitted",
        "wrong_context_factual_control": wrong_context["next_state_mse"] > 1e-5,
        "fresh_factual_control": fresh["hit_rate"] == 0.0,
        "persistence_exact": persisted["next_state_mse"] < 1e-6,
    }
    report = {
        "schema": "neural-computer.external-learned-evidence-admission-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "evidence_rows": EVIDENCE_ROWS,
            "evidence_updates": EVIDENCE_UPDATES,
            "noise_scale": NOISE_SCALE,
            "policy": "learned_transition_evidence_read_only_reuse_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "clean_trace": clean_trace,
        "learned_noisy_trace": learned_trace,
        "fixed_exact_trace": fixed_trace,
        "reversal_trace": reversal_trace,
        "noisy_probe": noisy_probe,
        "retained": retained,
        "wrong_context": wrong_context,
        "fresh": fresh,
        "persisted": persisted,
        "evidence_evaluator": {
            "optimizer_updates": EVIDENCE_UPDATES,
            "training_rows": EVIDENCE_ROWS * 2,
            "replayed_training_rows": EVIDENCE_ROWS * 2 * max(0, EVIDENCE_UPDATES - 1),
            "final_loss": evaluator_loss,
            "positive_probability": positive_probability,
            "negative_probability": negative_probability,
            "digest": evaluator.digest(),
        },
        "accounting": {
            "unique_transition_lifetimes": 3 + 3 + 2,
            "unique_verifier_bits": 3 + 3 + 2,
            "target_optimizer_updates": 0,
            "target_replayed_examples": 0,
            "memory_records_before_noise": source_record_count,
            "memory_records_after_noise_and_reversal": memory.record_count,
            "learned_context_addresses": resolver.context_count,
        },
        "controller_digest": controller_digest,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
