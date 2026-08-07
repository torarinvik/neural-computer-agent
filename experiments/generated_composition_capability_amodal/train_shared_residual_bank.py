"""Audit replay-free growth with a shared neural base and isolated residuals.

Each procedure gets a fresh external residual slot and decoder.  After the
first procedure is learned, the shared context encoder and all protected old
slots are frozen.  Later procedures receive only fresh verifier outcomes for
the current procedure plus a fresh forward auxiliary stream.  This measures
whether shared computation reduces growth cost without allowing a dense rewrite
to silently erase earlier capabilities.

This is a bounded shared-computation audit, not a claim of general continual
learning or arbitrary program induction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    generate_runtime_program_grammar,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _feedback,
    _runtime,
)
from neural_computer import (
    ExternalCapabilityProgram,
    ExternalCapabilityResidualComputeBank,
    ExternalCapabilitySharedResidualBank,
    OpaqueProtocolDecoder,
)

EVENT_WIDTH = 32
ACTION_WIDTH = 2
INTENTION_WIDTH = 16
CONTEXT_HIDDEN = 64
CONTEXT_WIDTH = 32
ADAPTER_HIDDEN = 64
DECODER_HIDDEN = 16
SPAN = 4
THRESHOLD = 0.75
CapabilityBank = ExternalCapabilitySharedResidualBank | ExternalCapabilityResidualComputeBank


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _digest_memory(
    bank: CapabilityBank,
    decoders: list[OpaqueProtocolDecoder],
) -> str:
    digest = hashlib.sha256()
    for prefix, module in (("bank", bank),):
        for name, value in sorted(module.state_dict().items()):
            digest.update(f"{prefix}.{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for decoder_index, decoder in enumerate(decoders):
        for name, value in sorted(decoder.state_dict().items()):
            digest.update(f"decoder.{decoder_index}.{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _rollout_slot(
    parent,
    bank: CapabilityBank,
    slot_index: int,
    decoder: OpaqueProtocolDecoder,
    batch,
    *,
    train: bool,
) -> dict[str, torch.Tensor]:
    device = batch.input_frames.device
    controller_state = parent.initial_state(batch.batch_size, device=device)
    capability_state = bank.initial_state(batch.batch_size, device=device).programs[
        slot_index
    ]
    zeros = torch.zeros(batch.batch_size, device=device)
    previous_action = torch.zeros(batch.batch_size, ACTION_WIDTH, device=device)
    previous_reward = zeros
    previous_propensity = torch.ones(batch.batch_size, device=device)
    previous_has_feedback = zeros
    present = torch.ones(batch.batch_size, dtype=torch.bool, device=device)
    quiet = _feedback(
        previous_action,
        previous_reward,
        previous_propensity,
        previous_has_feedback,
    )
    encoder = parent.encoders["vision"]

    def tick(frame: torch.Tensor, feedback):
        nonlocal controller_state, capability_state
        with torch.no_grad():
            event = encoder(frame)
            output, controller_state = parent.step_streams(
                {"vision": frame}, controller_state, feedback
            )
        adapted, capability_state = bank.step_slot(
            slot_index=slot_index,
            event=event,
            action=previous_action,
            outcome=previous_reward,
            intention=output.intention,
            state=capability_state,
            present=present,
        )
        return decoder(adapted)

    for frame in batch.input_frames.transpose(0, 1):
        tick(frame, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        tick(frame, quiet)

    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
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
        probabilities = torch.softmax(logits, dim=-1)
        if train:
            action = torch.multinomial(probabilities * 0.9 + 0.05, 1).squeeze(1)
        else:
            action = logits.argmax(dim=-1)
        reward = (action == correct).to(logits.dtype)
        selected = logits.gather(1, action.unsqueeze(1)).squeeze(1)
        losses.append(F.binary_cross_entropy_with_logits(selected, reward))
        rewards.append(reward)
        previous_action = F.one_hot(action, ACTION_WIDTH).to(logits.dtype)
        previous_reward = reward
        previous_propensity = probabilities.gather(
            1, action.unsqueeze(1)
        ).squeeze(1).detach()
        previous_propensity = previous_propensity.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        previous_has_feedback = torch.ones_like(previous_reward)
    return {"loss": torch.stack(losses).mean(), "rewards": torch.stack(rewards, dim=1)}


def _batch(
    count: int,
    *,
    span: int,
    seed: int,
    program_id: int,
    grammar,
):
    return generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation="generated_composition",
        generated_composition_ids=(program_id,),
        generated_compositions=grammar,
    )


def _train_slot(
    parent,
    bank: CapabilityBank,
    slot_index: int,
    decoder: OpaqueProtocolDecoder,
    program_id: int,
    grammar,
    *,
    updates: int,
    batch_size: int,
    audit_count: int,
    eval_every: int,
    seed: int,
    learning_rate: float,
) -> list[dict[str, float | int]]:
    trainable = [parameter for parameter in bank.parameters() if parameter.requires_grad]
    trainable += list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    progress: list[dict[str, float | int]] = []
    bank.train()
    decoder.train()
    for update in range(1, updates + 1):
        target = _batch(
            batch_size,
            span=SPAN,
            seed=seed + update * 10_007,
            program_id=program_id,
            grammar=grammar,
        )
        target_result = _rollout_slot(
            parent, bank, slot_index, decoder, target, train=True
        )
        auxiliary = generate_sequence_memory_batch(
            batch_size,
            span=2,
            distractors=1,
            seed=seed + 5_000_003 + update * 20_021,
            operation="forward",
        )
        auxiliary_result = _rollout_slot(
            parent, bank, slot_index, decoder, auxiliary, train=True
        )
        loss = target_result["loss"] + auxiliary_result["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update == updates or update % eval_every == 0:
            bank.eval()
            decoder.eval()
            heldout = _batch(
                audit_count,
                span=SPAN,
                seed=seed + 1_000_000 + update,
                program_id=program_id,
                grammar=grammar,
            )
            progress.append(
                {
                    "update": update,
                    "unique_verifier_bits": update * batch_size * SPAN,
                    "heldout_accuracy": float(
                        _rollout_slot(
                            parent, bank, slot_index, decoder, heldout, train=False
                        )["rewards"].mean()
                    ),
                }
            )
            bank.train()
            decoder.train()
    bank.eval()
    decoder.eval()
    return progress


@torch.no_grad()
def _accuracy(
    parent,
    bank: CapabilityBank,
    slot_index: int,
    decoder: OpaqueProtocolDecoder,
    program_id: int,
    grammar,
    *,
    count: int,
    seed: int,
) -> float:
    batch = _batch(
        count,
        span=SPAN,
        seed=seed,
        program_id=program_id,
        grammar=grammar,
    )
    return float(
        _rollout_slot(parent, bank, slot_index, decoder, batch, train=False)[
            "rewards"
        ].mean()
    )


def _probe_accuracy(
    parent,
    bank: ExternalCapabilitySharedResidualBank,
    slot_index: int,
    decoder: OpaqueProtocolDecoder,
    program_id: int,
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> tuple[float, list[float]]:
    outcomes = [
        _accuracy(
            parent,
            bank,
            slot_index,
            decoder,
            program_id,
            grammar,
            count=count,
            seed=seed + probe * 101,
        )
        for probe in range(probes)
    ]
    return min(outcomes), outcomes


def _stable_bits(progress: list[dict[str, float | int]], *, batch_size: int) -> int | None:
    for index, row in enumerate(progress):
        if all(
            float(later["heldout_accuracy"]) >= THRESHOLD
            for later in progress[index:]
        ):
            return int(row["update"]) * batch_size * SPAN
    return None


def _new_decoder(seed: int) -> OpaqueProtocolDecoder:
    torch.manual_seed(seed)
    return OpaqueProtocolDecoder(INTENTION_WIDTH, ACTION_WIDTH, hidden=DECODER_HIDDEN)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.slot_updates,
        args.batch_size,
        args.audit_count,
        args.eval_every,
        args.retention_probes,
    ) < 1:
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    source_ids = tuple(args.source_ids)
    if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be distinct and contain at least two IDs")
    grammar = generate_runtime_program_grammar(
        seed=args.program_seed,
        count=args.program_count,
        depth=args.program_depth,
        primitive_family=args.primitive_family,
    )
    if any(program_id < 0 or program_id >= len(grammar) for program_id in source_ids):
        raise ValueError("source ID is out of range")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    _parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())

    if args.residual_compute:
        bank: CapabilityBank = ExternalCapabilityResidualComputeBank(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            slot_count=1,
            shared_context_hidden=CONTEXT_HIDDEN,
            shared_context_width=CONTEXT_WIDTH,
            residual_context_hidden=args.residual_context_hidden,
            residual_context_width=args.residual_context_width,
            adapter_hidden=ADAPTER_HIDDEN,
        )
    else:
        bank = ExternalCapabilitySharedResidualBank(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            slot_count=1,
            context_hidden=CONTEXT_HIDDEN,
            context_width=CONTEXT_WIDTH,
            adapter_hidden=ADAPTER_HIDDEN,
        )
    decoders = [_new_decoder(args.seed + 10_000)]
    stage_records: list[dict[str, object]] = []
    progress_by_slot: list[list[dict[str, float | int]]] = []
    behavior_by_slot: list[float] = []
    old_slot_digests: list[dict[int, str]] = []
    shared_base_digest_after_first: str | None = None

    for stage_index, program_id in enumerate(source_ids):
        if stage_index:
            old_slot_digests.append(
                {
                    old_slot: _digest_module(bank.residual_slots[old_slot])
                    for old_slot in range(bank.slot_count)
                }
            )
            bank.freeze_shared_base()
            for old_slot in range(bank.slot_count):
                bank.freeze_slot(old_slot)
            slot_index = bank.add_slot()
            decoders.append(_new_decoder(args.seed + 10_000 + stage_index))
        else:
            slot_index = 0
        progress = _train_slot(
            parent,
            bank,
            slot_index,
            decoders[slot_index],
            program_id,
            grammar,
            updates=args.slot_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 20_000 + stage_index * 10_003,
            learning_rate=args.learning_rate,
        )
        progress_by_slot.append(progress)
        if stage_index == 0:
            shared_base_digest_after_first = _digest_module(
                bank.shared_context_encoder
            )
        behavior, behavior_probes = _probe_accuracy(
            parent,
            bank,
            slot_index,
            decoders[slot_index],
            program_id,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 40_000 + stage_index * 10_003,
        )
        behavior_by_slot.append(behavior)
        retained: list[float] = []
        retained_probes: list[list[float]] = []
        for old_slot in range(stage_index):
            retained_behavior, old_probe_outcomes = _probe_accuracy(
                parent,
                bank,
                old_slot,
                decoders[old_slot],
                source_ids[old_slot],
                grammar,
                count=args.audit_count,
                probes=args.retention_probes,
                seed=args.seed + 50_000 + stage_index * 10_003 + old_slot,
            )
            retained.append(retained_behavior)
            retained_probes.append(old_probe_outcomes)
        stage_records.append(
            {
                "stage": stage_index,
                "program_id": program_id,
                "slot_index": slot_index,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in bank.parameters()
                    if parameter.requires_grad
                )
                + _parameter_count(decoders[slot_index]),
                "stable_bits_to_threshold": _stable_bits(
                    progress, batch_size=args.batch_size
                ),
                "new_slot_behavior": behavior,
                "new_slot_probe_outcomes": behavior_probes,
                "retained_old_slot_behavior": retained,
                "retained_old_slot_probe_outcomes": retained_probes,
                "fresh_outcomes_only": True,
            }
        )

    old_slot_digests_unchanged = all(
        digest == _digest_module(bank.residual_slots[slot_index])
        for stage_digests in old_slot_digests
        for slot_index, digest in stage_digests.items()
    )
    final_behaviors: list[float] = []
    final_probe_outcomes: list[list[float]] = []
    for slot_index, program_id in enumerate(source_ids):
        final_behavior, probe_outcomes = _probe_accuracy(
            parent,
            bank,
            slot_index,
            decoders[slot_index],
            program_id,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 60_000 + slot_index,
        )
        final_behaviors.append(final_behavior)
        final_probe_outcomes.append(probe_outcomes)
    base_digest_before_reload = _digest_module(bank.shared_context_encoder)
    shared_base_stable_after_growth = (
        shared_base_digest_after_first is not None
        and shared_base_digest_after_first == base_digest_before_reload
    )
    bank_state = {name: value.detach().clone() for name, value in bank.state_dict().items()}
    decoder_states = [
        {name: value.detach().clone() for name, value in decoder.state_dict().items()}
        for decoder in decoders
    ]
    if args.residual_compute:
        reload_bank: CapabilityBank = ExternalCapabilityResidualComputeBank(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            slot_count=len(source_ids),
            shared_context_hidden=CONTEXT_HIDDEN,
            shared_context_width=CONTEXT_WIDTH,
            residual_context_hidden=args.residual_context_hidden,
            residual_context_width=args.residual_context_width,
            adapter_hidden=ADAPTER_HIDDEN,
        )
    else:
        reload_bank = ExternalCapabilitySharedResidualBank(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            slot_count=len(source_ids),
            context_hidden=CONTEXT_HIDDEN,
            context_width=CONTEXT_WIDTH,
            adapter_hidden=ADAPTER_HIDDEN,
        )
    reload_bank.load_state_dict(bank_state, strict=True)
    reload_decoders = [_new_decoder(args.seed + 10_000 + index) for index in range(len(source_ids))]
    for decoder, state in zip(reload_decoders, decoder_states, strict=True):
        decoder.load_state_dict(state, strict=True)
        decoder.eval()
    reload_exact = all(
        torch.equal(value, reload_bank.state_dict()[name])
        for name, value in bank_state.items()
    ) and all(
        torch.equal(value, reload_decoders[index].state_dict()[name])
        for index, state in enumerate(decoder_states)
        for name, value in state.items()
    )
    reload_behaviors: list[float] = []
    reload_probe_outcomes: list[list[float]] = []
    for slot_index, program_id in enumerate(source_ids):
        reload_behavior, probe_outcomes = _probe_accuracy(
            parent,
            reload_bank,
            slot_index,
            reload_decoders[slot_index],
            program_id,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 60_000 + slot_index,
        )
        reload_behaviors.append(reload_behavior)
        reload_probe_outcomes.append(probe_outcomes)
    clean_memory_digest = _digest_memory(bank, decoders)
    corruption_slot = len(source_ids) - 1
    for parameter in bank.residual_slots[corruption_slot].parameters():
        with torch.no_grad():
            parameter.zero_()
            parameter.reshape(-1)[0] = 100.0
    for parameter in decoders[corruption_slot].parameters():
        with torch.no_grad():
            parameter.zero_()
            parameter.reshape(-1)[0] = 100.0
    corrupted_behavior, corrupted_probe_outcomes = _probe_accuracy(
        parent,
        bank,
        corruption_slot,
        decoders[corruption_slot],
        source_ids[corruption_slot],
        grammar,
        count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 70_000 + corruption_slot,
    )
    corrupted_memory_digest = _digest_memory(bank, decoders)
    bank.load_state_dict(bank_state, strict=True)
    decoders[corruption_slot].load_state_dict(
        decoder_states[corruption_slot], strict=True
    )
    recovered_behavior, recovered_probe_outcomes = _probe_accuracy(
        parent,
        bank,
        corruption_slot,
        decoders[corruption_slot],
        source_ids[corruption_slot],
        grammar,
        count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 70_000 + corruption_slot,
    )
    recovered_memory_digest = _digest_memory(bank, decoders)
    parent_digest_after = _digest_core(parent, ())
    full_program = ExternalCapabilityProgram(
        EVENT_WIDTH,
        ACTION_WIDTH,
        INTENTION_WIDTH,
        context_hidden=CONTEXT_HIDDEN,
        context_width=CONTEXT_WIDTH,
        adapter_hidden=ADAPTER_HIDDEN,
    )
    full_payload = _parameter_count(full_program) + _parameter_count(decoders[0])
    shared_payload = _parameter_count(bank) + sum(
        _parameter_count(decoder) for decoder in decoders
    )
    report = {
        "schema": "neural-computer.generated-composition-shared-residual-bank-report.v1",
        "claim_boundary": (
            "A frozen controller acquired a finite sequence of opaque runtime "
            "procedures. One shared context encoder was trained on the first "
            "procedure, then frozen; each later procedure trained only an "
            "isolated residual slot and decoder from fresh outcomes. This is "
            "bounded shared-computation growth, not general continual learning."
        ),
        "seed": args.seed,
        "source_ids": list(source_ids),
        "bank_mode": (
            "shared_base_plus_residual_compute"
            if args.residual_compute
            else "shared_base_plus_intention_adapter"
        ),
        "programs": [list(grammar[program_id]) for program_id in source_ids],
        "budgets": {
            "parent_updates": args.parent_updates,
            "slot_updates": args.slot_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "retention_probes": args.retention_probes,
            "eval_every": args.eval_every,
            "torch_threads": args.torch_threads,
        },
        "stages": stage_records,
        "final_behavior": final_behaviors,
        "final_probe_outcomes": final_probe_outcomes,
        "reload_behavior": reload_behaviors,
        "reload_probe_outcomes": reload_probe_outcomes,
        "memory_corruption": {
            "slot": corruption_slot,
            "corrupted_behavior": corrupted_behavior,
            "corrupted_probe_outcomes": corrupted_probe_outcomes,
            "recovered_behavior": recovered_behavior,
            "recovered_probe_outcomes": recovered_probe_outcomes,
            "clean_digest": clean_memory_digest,
            "corrupted_digest": corrupted_memory_digest,
            "recovered_digest": recovered_memory_digest,
            "checksum_mismatch_detected": (
                corrupted_memory_digest != clean_memory_digest
            ),
        },
        "parameter_accounting": {
            "full_program_plus_decoder_per_slot": full_payload,
            "shared_bank_plus_decoder_total": shared_payload,
            "ratio_to_independent_slots": shared_payload
            / (len(source_ids) * full_payload),
            "shared_context_encoder": _parameter_count(bank.shared_context_encoder),
            "residual_per_slot": _parameter_count(bank.residual_slots[0]),
            "decoder_per_slot": _parameter_count(decoders[0]),
        },
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": args.parent_updates * args.batch_size * 2
            + len(source_ids) * args.slot_updates * args.batch_size * (SPAN + 2),
            "unique_logical_lifetimes": args.parent_updates * args.batch_size
            + len(source_ids) * args.slot_updates * args.batch_size * 2,
            "optimizer_updates": args.parent_updates
            + len(source_ids) * args.slot_updates,
            "replayed_examples": 0,
            "retention_observations": (
                sum(range(1, len(source_ids) + 1)) * args.retention_probes
                + 2 * len(source_ids) * args.retention_probes
            ),
        },
        "controls": {
            "old_slot_weights_unchanged": old_slot_digests_unchanged,
            "shared_base_frozen_after_first_slot": all(
                not parameter.requires_grad
                for parameter in bank.shared_context_encoder.parameters()
            ),
            "shared_base_digest_stable_after_growth": shared_base_stable_after_growth,
            "final_reload_exact": reload_exact,
            "memory_corruption_detected_and_recovered": (
                corrupted_memory_digest != clean_memory_digest
                and recovered_memory_digest == clean_memory_digest
                and recovered_behavior >= THRESHOLD
            ),
            "parent_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
        "gates": {
            "parent_stable": _stable_bits(parent_progress, batch_size=args.batch_size)
            is not None,
            "all_slots_stable": all(
                _stable_bits(progress, batch_size=args.batch_size) is not None
                for progress in progress_by_slot
            ),
            "all_slots_mastered": min(final_behaviors, default=0.0) >= THRESHOLD,
            "retained_during_growth": all(
                all(value >= THRESHOLD for value in stage["retained_old_slot_behavior"])
                for stage in stage_records
            ),
            "retained_after_reload": min(reload_behaviors, default=0.0) >= THRESHOLD,
            "old_slot_weights_unchanged": old_slot_digests_unchanged,
            "shared_base_frozen_after_first_slot": all(
                not parameter.requires_grad
                for parameter in bank.shared_context_encoder.parameters()
            ),
            "shared_base_digest_stable_after_growth": shared_base_stable_after_growth,
            "shared_payload_reduced": shared_payload < len(source_ids) * full_payload,
            "final_reload_exact": reload_exact,
            "memory_corruption_detected_and_recovered": (
                corrupted_memory_digest != clean_memory_digest
                and recovered_memory_digest == clean_memory_digest
                and recovered_behavior >= THRESHOLD
            ),
            "core_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
        "wall_seconds": perf_counter() - started,
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--program-seed", type=int, default=4242)
    parser.add_argument("--program-count", type=int, default=5)
    parser.add_argument("--program-depth", type=int, default=8)
    parser.add_argument(
        "--primitive-family", choices=("registry", "opaque_rule"), default="opaque_rule"
    )
    parser.add_argument("--source-ids", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--slot-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--residual-compute",
        action="store_true",
        help="give each appended slot a compact recurrent compute encoder",
    )
    parser.add_argument("--residual-context-hidden", type=int, default=16)
    parser.add_argument("--residual-context-width", type=int, default=8)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="set PyTorch intra-op threads for this tiny audit workload",
    )
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "final_behavior": report["final_behavior"],
                "reload_behavior": report["reload_behavior"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
