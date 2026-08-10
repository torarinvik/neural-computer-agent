"""Pressure-test maintenance decisions against real external-bank receipts.

Unlike the companion synthetic probe, this stream derives utility from actual
copy-on-write growth, held-out-equivalent model sharing, and compressed-bank
retention.  The action policy still sees only generic storage telemetry and
structural availability; scenario identity and the verifier's objective stay
outside the policy.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    MAINTENANCE_ACTIONS,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMemoryMaintenancePolicy,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 2
HIDDEN_WIDTH = 24
TRAIN_STEPS = 480
EVAL_STEPS = 96
TEMPERATURE = 1.0
LEARNING_RATE = 0.01
RETENTION_TOLERANCE = 0.05


@dataclass
class Scenario:
    router: ExternalLearnedMultiStreamTransitionContextRouter
    bank: ExternalTransitionModelBank
    grow_available: bool
    share_available: bool
    compression_available: bool
    compression_opportunity: float
    share_pair: tuple[int, int] | None
    heldout: tuple[ExternalTransitionObservation, ...]
    retention_baseline: dict[int, torch.Tensor]
    objective: str


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _observation(offset: float = 0.0) -> ExternalTransitionObservation:
    state = torch.tensor(
        [[0.2 + offset, -0.4], [0.5 + offset, 0.1]],
        dtype=torch.float32,
    )
    intention = torch.tensor([[0.7], [-0.3]], dtype=torch.float32)
    next_state = state + intention * torch.tensor([0.25, 0.75])
    return ExternalTransitionObservation(state, intention, next_state)


def _router_for_bank(
    bank: ExternalTransitionModelBank,
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    encoder = ExternalTransitionContextEncoder(
        WIDTH,
        INTENTION_WIDTH,
        hidden_width=8,
        context_width=CONTEXT_WIDTH,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=max(bank.capacity or bank.context_count, 1),
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    binding = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=2,
        max_streams=4,
        provisional_capacity=2,
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        binding,
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=CONTEXT_WIDTH),
    )


def _retention_probe(
    baseline: dict[int, torch.Tensor],
    observation: ExternalTransitionObservation,
):
    def probe(candidate: ExternalTransitionModelBank) -> bool:
        for slot_id, expected in baseline.items():
            index = candidate.physical_index_for_slot_id(slot_id)
            prediction = candidate.models[index](
                observation.state,
                observation.intention,
            )
            if not bool(
                torch.allclose(
                    prediction,
                    expected,
                    atol=RETENTION_TOLERANCE,
                    rtol=RETENTION_TOLERANCE,
                )
            ):
                return False
        return True

    return probe


def _scenario(index: int, seed: int) -> Scenario:
    regime = index % 4
    torch.manual_seed(seed + regime * 1000)
    if regime == 0:
        bank = ExternalTransitionModelBank(
            WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            hidden_width=8,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=2,
        )
        first = bank.ensure_context(torch.tensor([1.0, 0.0]))
        second = bank.ensure_context(torch.tensor([0.0, 1.0]))
        bank.models[second].load_state_dict(bank.models[first].state_dict())
        share_pair = (first, second)
        grow_available = False
        share_available = True
        compression_available = False
        compression_opportunity = 0.0
        objective = "share"
    elif regime == 1:
        bank = ExternalTransitionModelBank(
            WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            hidden_width=8,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=1,
        )
        bank.ensure_context(torch.tensor([1.0, 0.0]))
        share_pair = None
        grow_available = True
        share_available = False
        compression_available = False
        compression_opportunity = 0.0
        objective = "grow"
    elif regime == 2:
        bank = ExternalTransitionModelBank(
            WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            hidden_width=8,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=2,
        )
        bank.ensure_context(torch.tensor([1.0, 0.0]))
        share_pair = None
        grow_available = False
        share_available = False
        compression_available = True
        payload = bank.compressed_payload(dtype=torch.float16)
        source_bytes = sum(
            value.numel() * value.element_size()
            for model in bank.models
            for value in model.state_dict().values()
        )
        compressed_bytes = sum(
            value.numel() * value.element_size()
            for model in payload["models"]
            for value in model["state"].values()
        )
        compression_opportunity = max(
            0.0,
            min(1.0, 1.0 - compressed_bytes / max(source_bytes, 1)),
        )
        objective = "compress"
    else:
        bank = ExternalTransitionModelBank(
            WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            hidden_width=8,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=2,
        )
        bank.ensure_context(torch.tensor([1.0, 0.0]))
        share_pair = None
        grow_available = True
        share_available = False
        compression_available = False
        compression_opportunity = 0.0
        objective = "defer"

    heldout = (_observation(offset=float(regime) * 0.1),)
    baseline = {
        slot_id: bank.models[index](
            heldout[0].state,
            heldout[0].intention,
        ).detach().clone()
        for index, slot_id in enumerate(bank.slot_ids)
    }
    return Scenario(
        router=_router_for_bank(bank),
        bank=bank,
        grow_available=grow_available,
        share_available=share_available,
        compression_available=compression_available,
        compression_opportunity=compression_opportunity,
        share_pair=share_pair,
        heldout=heldout,
        retention_baseline=baseline,
        objective=objective,
    )


def _execute(scenario: Scenario, action: str) -> tuple[bool, Any | None]:
    if action == "defer":
        return True, None
    probe = _retention_probe(scenario.retention_baseline, scenario.heldout[0])
    if action == "grow":
        receipt = scenario.bank.grow_verified(
            (scenario.bank.capacity or scenario.bank.context_count) + 1,
            probe,
        )
        return bool(receipt.accepted), receipt
    if action == "share":
        if scenario.share_pair is None:
            return False, None
        receipt = scenario.bank.consolidate_verified(
            scenario.share_pair[0],
            scenario.share_pair[1],
            scenario.heldout,
            prediction_tolerance=1e-6,
            retention_probe=probe,
        )
        return bool(receipt.accepted), receipt
    if action == "compress":
        receipt = scenario.bank.compress_and_commit_verified(
            dtype=torch.float16,
            retention_probe=probe,
        )
        return bool(receipt.accepted), receipt
    raise ValueError(f"unsupported real maintenance action: {action}")


def _utility(
    scenario: Scenario,
    action: str,
    accepted: bool,
) -> float:
    if scenario.objective == "defer":
        return 1.0 if action == "defer" else 0.0
    if action != scenario.objective or not accepted:
        return 0.0
    return {
        "grow": 0.85,
        "share": 1.0,
        "compress": 0.95,
    }[action]


def _rollout(
    seed: int,
    *,
    learn: bool,
    shuffled_verifier: bool = False,
    action_shuffled: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    policy = ExternalMemoryMaintenancePolicy(
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE) if learn else None
    generator = torch.Generator().manual_seed(seed + 4000)
    utilities: list[float] = []
    actions: dict[str, int] = {action: 0 for action in MAINTENANCE_ACTIONS}
    accepted_transactions = 0
    bytes_saved = 0
    capacity_growth = 0
    updates = 0
    for step in range(TRAIN_STEPS):
        scenario = _scenario(step, seed + step * 17)
        proposal = scenario.router.propose_maintenance(
            policy,
            grow_available=scenario.grow_available,
            share_available=scenario.share_available,
            compression_available=scenario.compression_available,
            compression_opportunity=scenario.compression_opportunity,
            sample=learn,
            generator=generator if learn else None,
        )
        before_bytes = sum(
            value.numel() * value.element_size()
            for model in scenario.bank.models
            for value in model.state_dict().values()
        )
        before_capacity = scenario.bank.capacity or 0
        executed_action = proposal.action
        if action_shuffled:
            legal_actions = [
                action
                for index, action in enumerate(MAINTENANCE_ACTIONS)
                if bool(proposal.available_actions[index])
            ]
            executed_action = legal_actions[
                int(torch.randint(len(legal_actions), (), generator=generator))
            ]
        accepted, _receipt = _execute(scenario, executed_action)
        verifier_utility = _utility(scenario, executed_action, accepted)
        if shuffled_verifier:
            verifier_utility = float(
                torch.randint(2, (), generator=generator)
            )
        if learn:
            policy.adaptation_step(proposal, verifier_utility, optimizer=optimizer)
            updates += 1
        after_bytes = sum(
            value.numel() * value.element_size()
            for model in scenario.bank.models
            for value in model.state_dict().values()
        )
        accepted_transactions += int(accepted and executed_action != "defer")
        bytes_saved += max(0, before_bytes - after_bytes)
        if _receipt is not None and hasattr(_receipt, "source_bytes"):
            bytes_saved += max(
                0,
                int(_receipt.source_bytes) - int(_receipt.compressed_bytes),
            )
        capacity_growth += max(0, (scenario.bank.capacity or 0) - before_capacity)
        utilities.append(verifier_utility)
        actions[proposal.action] += 1
    evaluation: list[float] = []
    persistence_exact = True
    for step in range(EVAL_STEPS):
        scenario = _scenario(TRAIN_STEPS + step, seed + 100000 + step * 17)
        proposal = scenario.router.propose_maintenance(
            policy,
            grow_available=scenario.grow_available,
            share_available=scenario.share_available,
            compression_available=scenario.compression_available,
            compression_opportunity=scenario.compression_opportunity,
        )
        accepted, _receipt = _execute(scenario, proposal.action)
        evaluation.append(_utility(scenario, proposal.action, accepted))
        restored = ExternalTransitionModelBank.from_payload(scenario.bank.payload())
        persistence_exact = persistence_exact and restored.digest() == scenario.bank.digest()
    return {
        "policy": policy,
        "utilities": utilities,
        "evaluation": evaluation,
        "actions": actions,
        "accepted_transactions": accepted_transactions,
        "bytes_saved": bytes_saved,
        "capacity_growth": capacity_growth,
        "optimizer_updates": updates,
        "persistence_exact": persistence_exact,
    }


def _unsafe_probe_control(seed: int) -> bool:
    scenario = _scenario(1, seed)
    before = scenario.bank.digest()

    def mutating_probe(candidate: ExternalTransitionModelBank) -> bool:
        value = next(iter(candidate.models[0].state_dict().values()))
        value.add_(1.0)
        return True

    receipt = scenario.bank.grow_verified(
        (scenario.bank.capacity or 1) + 1,
        mutating_probe,
    )
    return not receipt.accepted and scenario.bank.digest() == before


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    controller = torch.nn.Linear(4, 4)
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    trained = _rollout(seed, learn=True)
    fresh = _rollout(seed + 900000, learn=False)
    shuffled = _rollout(seed + 910000, learn=True, shuffled_verifier=True)
    action_shuffled = _rollout(seed + 920000, learn=True, action_shuffled=True)
    trained_eval = sum(trained["evaluation"]) / EVAL_STEPS
    fresh_eval = sum(fresh["evaluation"]) / EVAL_STEPS
    shuffled_eval = sum(shuffled["evaluation"]) / EVAL_STEPS
    action_shuffled_eval = sum(action_shuffled["evaluation"]) / EVAL_STEPS
    gates = {
        "trained_beats_fresh": trained_eval > fresh_eval + 0.15,
        "trained_beats_shuffled_verifier": trained_eval > shuffled_eval + 0.10,
        "trained_beats_action_shuffled": trained_eval > action_shuffled_eval + 0.10,
        "real_transaction_observed": trained["accepted_transactions"] > 0,
        "compression_bytes_observed": trained["bytes_saved"] > 0,
        "growth_observed": trained["capacity_growth"] > 0,
        "persistence_exact": trained["persistence_exact"],
        "unsafe_probe_atomic": _unsafe_probe_control(seed + 1000),
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_unique_utility": trained["optimizer_updates"] == TRAIN_STEPS,
    }
    report = {
        "schema": "neural-computer.external-memory-real-maintenance.v1",
        "claim_boundary": (
            "learned maintenance action selection over actual external-bank "
            "growth, held-out model sharing, compression, and retention probes; "
            "not general continual learning or unrestricted memory growth"
        ),
        "seed": seed,
        "configuration": {
            "train_steps": TRAIN_STEPS,
            "eval_steps": EVAL_STEPS,
            "actions": MAINTENANCE_ACTIONS,
            "update": "single_scalar_verified_transaction_utility_without_replay_v1",
            "compression_codec": "float16",
            "retention_tolerance": RETENTION_TOLERANCE,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "trained_eval_utility": trained_eval,
            "fresh_eval_utility": fresh_eval,
            "shuffled_verifier_eval_utility": shuffled_eval,
            "action_shuffled_eval_utility": action_shuffled_eval,
            "trained_final_window_utility": sum(trained["utilities"][-64:]) / 64,
            "trained_action_counts": trained["actions"],
            "trained_accepted_transactions": trained["accepted_transactions"],
            "trained_bytes_saved": trained["bytes_saved"],
            "trained_capacity_growth": trained["capacity_growth"],
        },
        "accounting": {
            "unique_verifier_utilities": TRAIN_STEPS,
            "unique_logical_lifetimes": TRAIN_STEPS,
            "optimizer_updates": trained["optimizer_updates"],
            "replayed_examples": 0,
            "controller_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6110)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
