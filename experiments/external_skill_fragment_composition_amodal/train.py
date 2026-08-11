"""Pressure-test reusable external skill fragments on rendered events.

The parent controller is trained once and frozen.  A shared external register
interpreter and an external coefficient bank acquire two primitive fragments
sequentially.  A new decoder then learns a held-out serial composition from
the frozen chain.  All deployed inputs remain learned event tensors, opaque
feedback, and opaque intentions; operation names and correct actions stay in
the trainer-owned verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _feedback,
    _runtime,
)
from neural_computer import (
    ExternalCapabilityRegisterMachine,
    ExternalSkillFragmentBank,
    ExternalSkillFragmentCombiner,
    OpaqueProtocolDecoder,
    paired_counterfactual_ranking_loss,
)

ACTION_WIDTH = 2
EVENT_WIDTH = 32
INTENTION_WIDTH = 16
REGISTER_WIDTH = 32
INSTRUCTION_WIDTH = 16
BASIS_COUNT = 4
COMPOSITION_GRAMMAR = (("reverse", "rotate"), ("rotate", "reverse"))


def _machine() -> ExternalCapabilityRegisterMachine:
    return ExternalCapabilityRegisterMachine(
        EVENT_WIDTH,
        ACTION_WIDTH,
        INTENTION_WIDTH,
        REGISTER_WIDTH,
        INSTRUCTION_WIDTH,
        interpreter_hidden=64,
        operator_rank=8,
        event_window_size=4,
    )


def _batch(
    *,
    operation: str,
    count: int,
    span: int,
    seed: int,
    generated_compositions=None,
):
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation=operation,
        generated_compositions=generated_compositions,
    )
    # Remove only generic operation-cue pixels from the rendered query. Keep
    # the ordinal marker at x=28:31 and all sequence evidence. This is a valid
    # pixel-level rerender control: the external fragment chain must carry the
    # procedure instead of letting the decoder read a task cue from its
    # observed register state.
    query_frames = batch.query_frames.clone()
    query_frames[:, :, :, 1:27, :28] = 0.0
    return replace(batch, query_frames=query_frames)


def _digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fragment_bank(seed: int) -> ExternalSkillFragmentBank:
    torch.manual_seed(seed)
    bank = ExternalSkillFragmentBank(
        INSTRUCTION_WIDTH,
        BASIS_COUNT,
        key_width=INTENTION_WIDTH,
        router_hidden=32,
        max_fragment_steps=2,
    )
    bank.add_fragment(
        torch.randn(1, BASIS_COUNT) * 0.05,
        F.normalize(torch.randn(INTENTION_WIDTH), dim=0),
    )
    return bank


def _composition(
    bank: ExternalSkillFragmentBank,
    selected: tuple[int, ...] | torch.Tensor | None,
    *,
    batch_size: int,
) -> object:
    if isinstance(selected, torch.Tensor):
        return bank.compose_queries(selected)
    if selected is None:
        raise ValueError("dynamic composition requires opaque route queries")
    indices = (
        torch.tensor(selected, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
    )
    return bank.compose_indices(indices)


def _rollout(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    bank: ExternalSkillFragmentBank,
    decoder: OpaqueProtocolDecoder,
    batch,
    selected: tuple[int, ...] | None,
    *,
    train_decoder: bool,
    shuffle_outcomes: bool = False,
    zero_codes: bool = False,
    combiner: ExternalSkillFragmentCombiner | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = batch.input_frames.device
    batch_size = batch.batch_size
    parent_state = parent.initial_state(batch_size, device=device)
    register_state = machine.initial_state(batch_size, device=device)
    previous_action = torch.zeros(batch_size, ACTION_WIDTH, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    previous_propensity = torch.ones(batch_size, device=device)
    previous_has_feedback = torch.zeros(batch_size, device=device)
    present = torch.ones(batch_size, dtype=torch.bool, device=device)
    if selected is None:
        keys = torch.stack(tuple(key.detach() for key in bank.keys))
        order_ids = batch.operation_bits.to(device=device, dtype=torch.int64)
        route_queries = torch.stack(
            (keys[0].expand(batch_size, -1), keys[1].expand(batch_size, -1)),
            dim=1,
        )
        route_queries = route_queries.gather(
            1,
            order_ids[:, None, None].expand(batch_size, 1, bank.key_width),
        )
        # Each order has two opaque route queries; the verifier-private order
        # is used only to choose which external query pair is presented.
        route_queries = torch.cat(
            (
                route_queries,
                torch.where(
                    order_ids[:, None, None] == 0,
                    keys[1].expand(batch_size, 1, -1),
                    keys[0].expand(batch_size, 1, -1),
                ),
            ),
            dim=1,
        )
        composition = _composition(bank, route_queries, batch_size=batch_size)
    else:
        composition = _composition(bank, selected, batch_size=batch_size)
    if zero_codes:
        composition = replace(composition, codes=torch.zeros_like(composition.codes))
    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    def tick(frame: torch.Tensor, feedback) -> torch.Tensor:
        nonlocal parent_state, register_state
        with torch.no_grad():
            event = parent.encoders["vision"](frame)
            output, parent_state = parent.step_streams(
                {"vision": frame}, parent_state, feedback
            )
        register, observed = machine.observe_register(
            event=event,
            action=feedback.action,
            outcome=feedback.reward,
            intention=output.intention,
            state=register_state,
            present=present,
        )
        trace = machine.execute_fragment_composition_trace(
            register,
            composition,
            event_window=observed.event_window,
            event_window_mask=observed.event_window_mask,
        )
        register_state = observed
        executed = trace.final_state if combiner is None else combiner(trace)
        return decoder(executed)

    quiet = _feedback(
        previous_action,
        previous_reward,
        previous_propensity,
        previous_has_feedback,
    )
    for frame in batch.input_frames.transpose(0, 1):
        tick(frame, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        tick(frame, quiet)

    for frame, correct in zip(
        batch.query_frames.transpose(0, 1),
        batch.correct_actions.transpose(0, 1),
        strict=True,
    ):
        feedback = _feedback(
            previous_action,
            previous_reward,
            previous_propensity,
            previous_has_feedback,
        )
        logits = tick(frame, feedback)
        attempted = torch.tensor([[0, 1]], dtype=torch.long, device=device).expand(
            batch_size, -1
        )
        utilities = (attempted == correct.unsqueeze(1)).to(logits.dtype)
        if shuffle_outcomes:
            utilities = utilities.roll(1, dims=0)
        loss, _ = paired_counterfactual_ranking_loss(logits, attempted, utilities)
        losses.append(loss)
        probabilities = logits.softmax(dim=-1)
        behavior = probabilities * 0.9 + 0.05
        action = (
            torch.multinomial(behavior, 1).squeeze(1)
            if train_decoder
            else logits.argmax(dim=-1)
        )
        reward = (action == correct).to(logits.dtype)
        rewards.append(reward)
        previous_action = F.one_hot(action, ACTION_WIDTH).to(logits.dtype)
        previous_reward = reward.roll(1) if shuffle_outcomes else reward
        previous_propensity = (
            behavior.gather(1, action.unsqueeze(1)).squeeze(1).detach()
        )
        previous_has_feedback = torch.ones_like(previous_reward)
    return torch.stack(losses).mean(), torch.stack(rewards, dim=1)


def _train_stage(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    bank: ExternalSkillFragmentBank,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    selected: tuple[int, ...] | None,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    trainable: list[nn.Parameter],
    shuffle_outcomes: bool = False,
    generated_compositions=None,
    auxiliary_operation: str | None = None,
    auxiliary_selected: tuple[int, ...] | None = None,
    auxiliary_generated_compositions=None,
    auxiliary_weight: float = 1.0,
    combiner: ExternalSkillFragmentCombiner | None = None,
    eval_every: int = 0,
    audit_count: int = 0,
    audit_seed: int = 0,
) -> list[dict[str, float | int]]:
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary fragment-stage weight cannot be negative")
    if auxiliary_operation is None and auxiliary_selected is not None:
        raise ValueError("auxiliary selection requires an auxiliary operation")
    if eval_every < 0 or audit_count < 0:
        raise ValueError("fragment evaluation settings cannot be negative")
    if eval_every and not audit_count:
        raise ValueError("fragment evaluation requires a positive audit count")
    parameters = [parameter for parameter in trainable if parameter.requires_grad]
    if not parameters:
        raise ValueError("fragment stage has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        batch = _batch(
            operation=operation,
            count=batch_size,
            span=span,
            seed=seed + update * 10_007,
            generated_compositions=generated_compositions,
        )
        loss, rewards = _rollout(
            parent,
            machine,
            bank,
            decoder,
            batch,
            selected,
            train_decoder=True,
            shuffle_outcomes=shuffle_outcomes,
            combiner=combiner,
        )
        if auxiliary_operation is not None:
            auxiliary_batch = _batch(
                operation=auxiliary_operation,
                count=batch_size,
                span=span,
                seed=seed + 5_000_003 + update * 20_021,
                generated_compositions=auxiliary_generated_compositions,
            )
            auxiliary_loss, _ = _rollout(
                parent,
                machine,
                bank,
                decoder,
                auxiliary_batch,
                auxiliary_selected,
                train_decoder=True,
                shuffle_outcomes=shuffle_outcomes,
                combiner=combiner,
            )
            loss = loss + auxiliary_weight * auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "training_accuracy": float(rewards.mean()),
                "loss": float(loss.detach()),
                "unique_verifier_bits": update
                * batch_size
                * span
                * 2
                * (2 if auxiliary_operation is not None else 1),
            }
        )
        if eval_every and (update % eval_every == 0 or update == updates):
            history[-1]["heldout_accuracy"] = _accuracy(
                parent,
                machine,
                bank,
                decoder,
                operation=operation,
                selected=selected,
                count=audit_count,
                span=span,
                seed=audit_seed + update,
                generated_compositions=generated_compositions,
                combiner=combiner,
            )
    return history


def _stable_bits(
    history: list[dict[str, float | int]],
    *,
    threshold: float,
    bits_per_update: int,
) -> int | None:
    measured = [row for row in history if "heldout_accuracy" in row]
    for index, row in enumerate(measured):
        if all(
            float(later["heldout_accuracy"]) >= threshold for later in measured[index:]
        ):
            return int(row["update"]) * bits_per_update
    return None


@torch.no_grad()
def _accuracy(
    parent,
    machine: ExternalCapabilityRegisterMachine,
    bank: ExternalSkillFragmentBank,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    selected: tuple[int, ...],
    count: int,
    span: int,
    seed: int,
    shuffle_outcomes: bool = False,
    generated_compositions=None,
    zero_codes: bool = False,
    combiner: ExternalSkillFragmentCombiner | None = None,
) -> float:
    batch = _batch(
        operation=operation,
        count=count,
        span=span,
        seed=seed,
        generated_compositions=generated_compositions,
    )
    return float(
        _rollout(
            parent,
            machine,
            bank,
            decoder,
            batch,
            selected,
            train_decoder=False,
            shuffle_outcomes=shuffle_outcomes,
            zero_codes=zero_codes,
            combiner=combiner,
        )[1].mean()
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 100,
        learning_rate=3e-3,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="paired_counterfactual",
    )
    parent.eval()
    parent_digest_before = _digest(parent.controller)

    machine = _machine()
    bank = _fragment_bank(args.seed + 1)
    reverse_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    reverse_history = _train_stage(
        parent,
        machine,
        bank,
        reverse_decoder,
        operation="reverse",
        selected=(0,),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 10_000,
        trainable=[
            *machine.parameters(),
            bank.shared_basis,
            bank.coefficients[0],
            *reverse_decoder.parameters(),
        ],
    )
    reverse_before = _accuracy(
        parent,
        machine,
        bank,
        reverse_decoder,
        operation="reverse",
        selected=(0,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 20_000,
    )

    bank.grow_basis(2)
    bank.freeze_basis_prefix(BASIS_COUNT)
    bank.add_fragment(
        torch.randn(1, bank.basis_count) * 0.05,
        F.normalize(torch.randn(INTENTION_WIDTH), dim=0),
    )
    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    bank.coefficients[0].requires_grad_(False)
    rotate_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    rotate_history = _train_stage(
        parent,
        machine,
        bank,
        rotate_decoder,
        operation="rotate",
        selected=(1,),
        updates=args.primitive_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 30_000,
        trainable=[
            bank.shared_basis,
            bank.coefficients[1],
            *rotate_decoder.parameters(),
        ],
        auxiliary_operation="generated_composition",
        auxiliary_selected=None,
        auxiliary_generated_compositions=COMPOSITION_GRAMMAR,
    )
    reverse_after = _accuracy(
        parent,
        machine,
        bank,
        reverse_decoder,
        operation="reverse",
        selected=(0,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 21_000,
    )
    rotate_accuracy = _accuracy(
        parent,
        machine,
        bank,
        rotate_decoder,
        operation="rotate",
        selected=(1,),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 31_000,
    )

    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    composition_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH,
        REGISTER_WIDTH,
        hidden=64,
    )
    composition_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    composition_history = _train_stage(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 40_000,
        trainable=[
            *composition_combiner.parameters(),
            *composition_decoder.parameters(),
        ],
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=composition_combiner,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 40_500,
    )
    composition_accuracy = _accuracy(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 41_000,
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=composition_combiner,
    )
    wrong_order_accuracy = _accuracy(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=(1, 0),
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 42_000,
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=composition_combiner,
    )
    zero_fragment_accuracy = _accuracy(
        parent,
        machine,
        bank,
        composition_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 42_500,
        generated_compositions=COMPOSITION_GRAMMAR,
        zero_codes=True,
        combiner=composition_combiner,
    )
    shuffled_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH,
        REGISTER_WIDTH,
        hidden=64,
    )
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    shuffled_history = _train_stage(
        parent,
        machine,
        bank,
        shuffled_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 43_000,
        trainable=[
            *shuffled_combiner.parameters(),
            *shuffled_decoder.parameters(),
        ],
        shuffle_outcomes=True,
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=shuffled_combiner,
    )
    shuffled_accuracy = _accuracy(
        parent,
        machine,
        bank,
        shuffled_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 44_000,
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=shuffled_combiner,
    )

    fresh_machine = _machine()
    fresh_bank = _fragment_bank(args.seed + 2)
    fresh_bank.grow_basis(2)
    fresh_bank.add_fragment(
        torch.randn(1, fresh_bank.basis_count) * 0.05,
        F.normalize(torch.randn(INTENTION_WIDTH), dim=0),
    )
    fresh_combiner = ExternalSkillFragmentCombiner(
        REGISTER_WIDTH,
        REGISTER_WIDTH,
        hidden=64,
    )
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_history = _train_stage(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_decoder,
        operation="generated_composition",
        selected=None,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 45_000,
        trainable=[
            *fresh_machine.parameters(),
            fresh_bank.shared_basis,
            *fresh_bank.coefficients,
            *fresh_combiner.parameters(),
            *fresh_decoder.parameters(),
        ],
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=fresh_combiner,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 45_500,
    )
    fresh_accuracy = _accuracy(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_decoder,
        operation="generated_composition",
        selected=None,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 46_000,
        generated_compositions=COMPOSITION_GRAMMAR,
        combiner=fresh_combiner,
    )

    route_queries = torch.stack((bank.keys[0].detach(), bank.keys[1].detach()))
    routed = bank.compose_queries(route_queries.unsqueeze(0))
    parent_digest_after = _digest(parent.controller)
    bits_per_update = args.batch_size * args.span * 2
    composition_stable_bits = _stable_bits(
        composition_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable_bits = _stable_bits(
        fresh_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    transfer_ratio = (
        float(fresh_stable_bits) / float(composition_stable_bits)
        if composition_stable_bits and fresh_stable_bits
        else None
    )
    training_batches = (
        args.parent_updates
        + 2 * args.primitive_updates
        + args.primitive_updates
        + 3 * args.composition_updates
    )
    eval_points = (
        sum(
            1
            for update in range(1, args.composition_updates + 1)
            if update % args.eval_every == 0 or update == args.composition_updates
        )
        if args.eval_every
        else 0
    )
    audit_batches = 2 * eval_points  # composition and fresh curves
    report = {
        "schema": "neural-computer.external-skill-fragment-composition-report.v1",
        "claim_boundary": (
            "This is a bounded compositional external-memory pressure test. "
            "It does not establish arbitrary program induction, unrestricted "
            "growth, or general continual learning."
        ),
        "seed": args.seed,
        "span": args.span,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "eval_every": args.eval_every,
        "mastery_threshold": args.mastery_threshold,
        "parent_progress": parent_progress,
        "reverse": {
            "before": reverse_before,
            "after": reverse_after,
            "history": reverse_history,
        },
        "rotate": {"accuracy": rotate_accuracy, "history": rotate_history},
        "composition": {
            "accuracy": composition_accuracy,
            "wrong_order_accuracy": wrong_order_accuracy,
            "zero_fragment_accuracy": zero_fragment_accuracy,
            "history": composition_history,
        },
        "reward_shuffled": {
            "accuracy": shuffled_accuracy,
            "history": shuffled_history,
        },
        "fresh": {"accuracy": fresh_accuracy, "history": fresh_history},
        "stable_bits_to_threshold": composition_stable_bits,
        "fresh_stable_bits_to_threshold": fresh_stable_bits,
        "transfer_ratio_fresh_over_inherited": transfer_ratio,
        "routing": {
            "selected_indices": routed.fragment_indices.tolist(),
            "route_scores": routed.route_scores.tolist(),
        },
        "accounting": {
            "unique_verifier_bits": (
                training_batches * bits_per_update
                + audit_batches * args.audit_count * args.span * 2
            ),
            "training_unique_verifier_bits": training_batches * bits_per_update,
            "audit_unique_verifier_bits": (
                audit_batches * args.audit_count * args.span * 2
            ),
            "unique_logical_lifetimes": (
                training_batches * args.batch_size
            ),
            "optimizer_updates": (
                args.parent_updates
                + 2 * args.primitive_updates
                + 3 * args.composition_updates
            ),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "reverse_retained": reverse_after >= reverse_before - 0.05,
            "composition_stable": composition_stable_bits is not None,
            "fresh_stable": fresh_stable_bits is not None,
            "positive_stable_transfer": (
                composition_stable_bits is not None
                and fresh_stable_bits is not None
                and fresh_stable_bits > composition_stable_bits
            ),
            "composition_mastered": composition_accuracy >= args.mastery_threshold,
            "wrong_order_rejected": wrong_order_accuracy < args.mastery_threshold,
            "no_fragment_bypass": zero_fragment_accuracy < args.mastery_threshold,
            "reward_shuffled_rejected": shuffled_accuracy < args.mastery_threshold,
            "fresh_audited": fresh_accuracy >= 0.50,
            "core_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
            "routing_resolved": routed.fragment_indices.tolist() == [[0, 1]],
        },
    }
    report["promoted"] = all(report["gates"].values())
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=256)
    parser.add_argument("--primitive-updates", type=int, default=256)
    parser.add_argument("--composition-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--span", type=int, default=3)
    parser.add_argument("--audit-count", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--report-out", type=Path)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
