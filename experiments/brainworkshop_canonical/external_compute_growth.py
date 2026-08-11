"""Acquire a new computation in an isolated external register file.

This audit is the next pressure rung after fixed temporal relation readers. A
frozen canonical controller and event frontend feed a generic external
register machine. Each capability is an opaque instruction plus an
append-only event-window compute basis; the basis is trained only from the
verifier's scalar action outcomes. The source file is frozen before the
triplet-parity file is trained, so retention is tested without replaying the
source task.

The experiment deliberately selects the file explicitly. Route discovery is a
separate concern and is not allowed to hide a computation-acquisition failure.
No rule family, target bit, or correct action crosses the verifier boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from neural_computer import (
    ControllerFeedback,
    ExternalCapabilityRegisterMachine,
    ExternalRegisterComputeBasis,
    ExternalRegisterInstruction,
    IntentEvent,
    KeypressDecoder,
)

from .cross_family_rule_growth import RULES, CrossFamilyVerifier
from .runner import CanonicalBrainWorkshopAgent

EXTERNAL_COMPUTE_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-compute-growth.v1"
)
SOURCE_FAMILY = "symbol_parity"
TARGET_FAMILY = "triplet_parity"
TARGET_CUE = 7
ACTION_COUNT = 2
EVENT_WIDTH = 16
INTENTION_WIDTH = 8
REGISTER_WIDTH = 16
INSTRUCTION_WIDTH = 8
EVENT_WINDOW_SIZE = 4
ENCODER_SYMBOL_COUNT = 13
MASTERY_THRESHOLD = 0.80


@dataclass
class ComputeGrowthSystem:
    """One frozen-core runtime and its independently addressable files."""

    agent: CanonicalBrainWorkshopAgent
    machine: ExternalCapabilityRegisterMachine
    instructions: nn.ModuleList
    readouts: nn.ModuleList
    decoders: nn.ModuleList


def _digest(*modules: nn.Module) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{module_index}:{name}".encode())
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _basis(
    *, hidden: int = 32, event_window_size: int = EVENT_WINDOW_SIZE
) -> ExternalRegisterComputeBasis:
    if event_window_size < 1:
        raise ValueError("event window size must be positive")
    return ExternalRegisterComputeBasis(
        REGISTER_WIDTH,
        INSTRUCTION_WIDTH,
        hidden=hidden,
        event_width=EVENT_WIDTH,
        event_window_size=event_window_size,
        microsteps=2,
        event_read_mode="flattened_window",
        register_input_mode="event_window_only",
    )


def _build(
    seed: int,
    *,
    slot_count: int = 2,
    basis_hidden: int = 32,
    event_window_size: int = EVENT_WINDOW_SIZE,
) -> ComputeGrowthSystem:
    """Build an append-only bank of opaque files over one fixed interpreter."""

    if slot_count < 1:
        raise ValueError("external compute slot count must be positive")
    if basis_hidden < 1:
        raise ValueError("external compute basis hidden width must be positive")
    if event_window_size < 1:
        raise ValueError("external compute event window size must be positive")

    torch.manual_seed(seed)
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=ENCODER_SYMBOL_COUNT,
        n_back=2,
        event_width=EVENT_WIDTH,
        intention_width=INTENTION_WIDTH,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
    )
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    machine = ExternalCapabilityRegisterMachine(
        EVENT_WIDTH,
        ACTION_COUNT,
        INTENTION_WIDTH,
        REGISTER_WIDTH,
        INSTRUCTION_WIDTH,
        interpreter_hidden=32,
        operator_mode="factorized_low_rank",
        operator_rank=4,
        basis_slots=tuple(
            _basis(
                hidden=basis_hidden,
                event_window_size=event_window_size,
            )
            for _ in range(slot_count)
        ),
        basis_hidden=basis_hidden,
        basis_microsteps=2,
        basis_event_read_mode="flattened_window",
        basis_register_input_mode="event_window_only",
        event_window_size=event_window_size,
    )
    instructions = nn.ModuleList(
        ExternalRegisterInstruction(INSTRUCTION_WIDTH) for _ in range(slot_count)
    )
    readouts = nn.ModuleList(
        nn.Sequential(
            nn.Linear(REGISTER_WIDTH, 16),
            nn.GELU(),
            nn.Linear(16, INTENTION_WIDTH),
        )
        for _ in range(slot_count)
    )
    decoders = nn.ModuleList(
        KeypressDecoder(INTENTION_WIDTH, ACTION_COUNT, hidden=16)
        for _ in range(slot_count)
    )
    return ComputeGrowthSystem(agent, machine, instructions, readouts, decoders)


def _common_modules(system: ComputeGrowthSystem) -> tuple[nn.Module, ...]:
    machine = system.machine
    return (
        machine.input_encoder,
        machine.context_recurrent,
        machine.register_writer,
        machine.register_write_gate,
    )


def _slot_modules(system: ComputeGrowthSystem, slot: int) -> tuple[nn.Module, ...]:
    if not 0 <= slot < len(system.instructions):
        raise ValueError("external compute slot is outside the bank")
    return (
        system.machine.basis_slots[slot],
        system.instructions[slot],
        system.readouts[slot],
        system.decoders[slot],
    )


def _set_requires_grad(modules: tuple[nn.Module, ...], enabled: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def _parameters(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    if not result:
        raise ValueError("external compute stage has no trainable parameters")
    return result


def _episode(
    system: ComputeGrowthSystem,
    *,
    family: str,
    slot: int,
    cue_symbol: int = TARGET_CUE,
    batch_size: int,
    steps: int,
    seed: int,
    train: bool,
    reset_external_each_step: bool = False,
    entropy_weight: float = 0.0,
    credit_mode: str = "reinforce",
    shuffle_outcomes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run one fresh verifier lifetime through INPUT -> PROCESS -> OUTPUT."""

    if family not in RULES:
        raise ValueError("unsupported external compute audit family")
    if entropy_weight < 0.0:
        raise ValueError("entropy weight cannot be negative")
    if credit_mode not in {"reinforce", "attempted_bce"}:
        raise ValueError("unsupported external compute credit mode")
    verifier = CrossFamilyVerifier(
        family=family,
        batch_size=batch_size,
        steps=steps,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    verifier.reset()
    agent = system.agent
    machine = system.machine
    controller_state = agent.initial_state(batch_size, device="cpu")
    register_state = machine.initial_state(batch_size, device="cpu")
    feedback = agent.initial_feedback(batch_size, device="cpu")
    previous_action = torch.zeros(batch_size, ACTION_COUNT)
    log_probabilities: list[torch.Tensor] = []
    selected_logits: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    delivered_rewards: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []

    while not verifier.done:
        if reset_external_each_step:
            register_state = machine.initial_state(batch_size, device="cpu")
        with torch.no_grad():
            collection = agent.runtime.encode_streams(
                {"stimulus": verifier.observation()}
            )
            controller_output, controller_state = agent.runtime.step_events(
                collection, controller_state, feedback
            )
        executed, register_state = machine.read_execute_register(
            event=collection.payload[:, 0],
            action=previous_action,
            outcome=feedback.reward,
            intention=controller_output.intention,
            state=register_state,
            instructions=(system.instructions[slot],),
            basis_slots=(slot,),
        )
        intention = IntentEvent(system.readouts[slot](executed))
        logits = system.decoders[slot](intention)
        probabilities = logits.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=-1
        )
        action = (
            torch.multinomial(probabilities, 1).squeeze(-1)
            if train
            else logits.argmax(dim=-1)
        )
        selected_logits.append(
            logits.gather(1, action[:, None]).squeeze(1)
        )
        propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        scored = verifier.score(action)
        delivered_reward = (
            scored.reward.roll(1) if shuffle_outcomes else scored.reward
        )
        log_probabilities.append(propensity.clamp_min(1e-8).log())
        entropies.append(entropy)
        rewards.append(scored.reward)
        delivered_rewards.append(delivered_reward)
        eligible.append(scored.eligible)
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(action),
            reward=delivered_reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size),
        )
        previous_action = F.one_hot(action, ACTION_COUNT).to(torch.float32)

    reward_tensor = torch.stack(rewards, dim=1)
    delivered_reward_tensor = torch.stack(delivered_rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    log_probability_tensor = torch.stack(log_probabilities, dim=1)
    denominator = eligible_tensor.sum().clamp_min(1.0)
    accuracy = (reward_tensor * eligible_tensor).sum() / denominator
    if credit_mode == "attempted_bce":
        selected_logit_tensor = torch.stack(selected_logits, dim=1)
        loss = F.binary_cross_entropy_with_logits(
            selected_logit_tensor[eligible_tensor],
            delivered_reward_tensor[eligible_tensor],
        )
    else:
        loss = -(
            (
                (delivered_reward_tensor - 0.5).detach()
                * log_probability_tensor
                * eligible_tensor
            ).sum()
            / denominator
        )
    if entropy_weight:
        entropy_tensor = torch.stack(entropies, dim=1)
        loss = loss - entropy_weight * (
            (entropy_tensor * eligible_tensor).sum() / denominator
        )
    return loss, accuracy, int(eligible_tensor.sum().item())


def _train_stage(
    system: ComputeGrowthSystem,
    *,
    family: str,
    slot: int,
    cue_symbol: int = TARGET_CUE,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    learning_rate: float,
    entropy_weight: float = 0.0,
    credit_mode: str = "reinforce",
    shuffle_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    modules = _common_modules(system) + _slot_modules(system, slot)
    optimizer = torch.optim.Adam(_parameters(modules), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        torch.manual_seed(seed + update * 10_007)
        loss, accuracy, bits = _episode(
            system,
            family=family,
            slot=slot,
            cue_symbol=cue_symbol,
            batch_size=batch_size,
            steps=steps,
            seed=seed + update,
            train=True,
            entropy_weight=entropy_weight,
            credit_mode=credit_mode,
            shuffle_outcomes=shuffle_outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_parameters(modules), max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "eligible_accuracy": float(accuracy.detach()),
                "unique_verifier_bits": batch_size * bits // max(batch_size, 1),
                "replayed_examples": 0,
            }
        )
    return history


@torch.no_grad()
def _evaluate(
    system: ComputeGrowthSystem,
    *,
    family: str,
    slot: int,
    cue_symbol: int = TARGET_CUE,
    lifetimes: int,
    batch_size: int,
    steps: int,
    seed: int,
    reset_external_each_step: bool = False,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for lifetime in range(lifetimes):
        _, accuracy, bits = _episode(
            system,
            family=family,
            slot=slot,
            cue_symbol=cue_symbol,
            batch_size=batch_size,
            steps=steps,
            seed=seed + lifetime,
            train=False,
            reset_external_each_step=reset_external_each_step,
        )
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(accuracy),
                "unique_verifier_bits": batch_size * bits // max(batch_size, 1),
                "replayed_examples": 0,
            }
        )
    return rows


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _mean(rows: list[dict[str, float | int]]) -> float:
    if not rows:
        raise ValueError("cannot average an empty evaluation")
    return sum(float(row["accuracy"]) for row in rows) / len(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.fresh_updates,
        args.batch_size,
        args.steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("external compute budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    source_modules = _common_modules(system) + _slot_modules(system, 0)
    target_modules = _slot_modules(system, 1)
    _set_requires_grad(_common_modules(system) + _slot_modules(system, 0), True)
    _set_requires_grad(_slot_modules(system, 1), False)
    source_history = _train_stage(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=TARGET_CUE,
        updates=args.source_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
    )
    source_before_target = _evaluate(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10_000,
    )
    source_slot_before = _digest(*source_modules)

    _set_requires_grad(source_modules, False)
    _set_requires_grad(target_modules, True)
    target_history = _train_stage(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        updates=args.target_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        learning_rate=args.learning_rate,
    )
    source_after_target = _evaluate(
        system,
        family=SOURCE_FAMILY,
        slot=0,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10_000,
    )
    target_retention = _evaluate(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000,
    )
    no_history_target = _evaluate(
        system,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 40_000,
        reset_external_each_step=True,
    )

    fresh = _build(args.seed + 50_000)
    # The comparator receives the same fresh target-file initialization but
    # cannot train the shared register path.  In event-window-only mode that
    # path is intentionally irrelevant, making this a fair file-level
    # comparison rather than a second controller-training condition.
    _set_requires_grad(_common_modules(fresh), False)
    _set_requires_grad(_slot_modules(fresh, 1), True)
    _set_requires_grad(_slot_modules(fresh, 0), False)
    _train_stage(
        fresh,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        updates=args.fresh_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 60_000,
        learning_rate=args.learning_rate,
    )
    fresh_target = _evaluate(
        fresh,
        family=TARGET_FAMILY,
        slot=1,
        cue_symbol=TARGET_CUE,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 70_000,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "source_mastered_before_growth": _stable(source_before_target),
        "target_mastered_after_growth": _stable(target_retention),
        "source_retained_after_growth": _stable(source_after_target),
        "source_outputs_unchanged_on_matched_lifetimes": source_before_target
        == source_after_target,
        "source_file_unchanged": source_slot_before == _digest(*source_modules),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "history_is_required": _mean(no_history_target) < 0.65,
        "zero_replayed_examples": True,
    }
    unique_training_bits = args.batch_size * (
        args.source_updates * (args.steps - 0)
        + args.target_updates * (args.steps - 3)
    )
    report = {
        "schema": EXTERNAL_COMPUTE_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-only acquisition of one new rendered temporal computation "
            "in an isolated external register file with frozen controller and "
            "frontend; this is not route discovery, unrestricted memory growth, "
            "or general continual learning."
        ),
        "architecture": {
            "boundary": "rendered_event -> frozen_amodal_controller -> external_register_file -> keypress_decoder",
            "external_file": "opaque_instruction_plus_append_only_event_window_compute_basis_v1",
            "source_family": SOURCE_FAMILY,
            "target_family": TARGET_FAMILY,
            "target_cue": TARGET_CUE,
            "event_window_size": EVENT_WINDOW_SIZE,
            "basis_microsteps": 2,
            "route": "explicit_external_slot_selection_for_computation_isolation",
            "weight_policy": (
                "source_file_frozen_target_file_initialized_fresh_shared_register_path_excluded"
            ),
        },
        "seed": args.seed,
        "source_history_tail": source_history[-5:],
        "target_history_tail": target_history[-5:],
        "evaluation": {
            "source_before_target": source_before_target,
            "source_after_target": source_after_target,
            "target_after_growth": target_retention,
            "target_without_external_history": no_history_target,
            "fresh_target_comparator": fresh_target,
            "transferred_target_mean": _mean(target_retention),
            "fresh_target_mean": _mean(fresh_target),
            "transfer_ratio": _mean(target_retention) / max(_mean(fresh_target), 1e-8),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_training_bits,
            "unique_logical_lifetimes": args.batch_size
            * (args.source_updates + args.target_updates),
            "optimizer_updates": args.source_updates + args.target_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": (
                unique_training_bits if gates["target_mastered_after_growth"] else None
            ),
            "retention_threshold": MASTERY_THRESHOLD,
        },
        "status": "promoted_external_computation_growth" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--fresh-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
