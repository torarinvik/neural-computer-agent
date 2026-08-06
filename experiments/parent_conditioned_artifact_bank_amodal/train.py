"""Route independently learned external recurrent programs through one bank."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
    _digest_core,
    _feedback,
    _runtime,
)
from neural_computer import (
    ControllerFeedback,
    ExecutableArtifactMemory,
    ExternalCapabilityProgram,
    OpaqueAddressRouter,
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
    paired_counterfactual_ranking_loss,
)

PROGRAMS = (
    ("reverse4", "reverse", 4),
    ("forward4", "forward", 4),
    ("complement4", "complement", 4),
)
EVENT_WIDTH = 32
ACTION_WIDTH = 2
INTENTION_WIDTH = 16
CAPABILITY_CONTEXT_HIDDEN = 64
CAPABILITY_CONTEXT_WIDTH = 32
CAPABILITY_ADAPTER_HIDDEN = 64
DECODER_HIDDEN = 16


def _stable_bits(
    progress: list[dict[str, float | int]],
    *,
    threshold: float,
    bits_per_update: int,
) -> int | None:
    for index, row in enumerate(progress):
        if all(
            float(later["heldout_accuracy"]) >= threshold
            for later in progress[index:]
        ):
            return int(row["update"]) * bits_per_update
    return None


def _new_capability(seed: int) -> tuple[ExternalCapabilityProgram, OpaqueProtocolDecoder]:
    torch.manual_seed(seed)
    return (
        ExternalCapabilityProgram(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            context_hidden=CAPABILITY_CONTEXT_HIDDEN,
            context_width=CAPABILITY_CONTEXT_WIDTH,
            adapter_hidden=CAPABILITY_ADAPTER_HIDDEN,
        ),
        OpaqueProtocolDecoder(INTENTION_WIDTH, ACTION_WIDTH, hidden=DECODER_HIDDEN),
    )


def _artifact(
    program: ExternalCapabilityProgram,
    decoder: OpaqueProtocolDecoder,
) -> dict[str, torch.Tensor]:
    return {
        **{
            f"program.{name}": value.detach().cpu().clone()
            for name, value in program.state_dict().items()
        },
        **{
            f"decoder.{name}": value.detach().cpu().clone()
            for name, value in decoder.state_dict().items()
        },
    }


def _load_artifact(
    artifact: dict[str, torch.Tensor],
) -> tuple[ExternalCapabilityProgram, OpaqueProtocolDecoder]:
    program, decoder = _new_capability(seed=0)
    program_state = {
        name.removeprefix("program."): value
        for name, value in artifact.items()
        if name.startswith("program.")
    }
    decoder_state = {
        name.removeprefix("decoder."): value
        for name, value in artifact.items()
        if name.startswith("decoder.")
    }
    if not program_state or not decoder_state:
        raise ValueError("capability artifact is missing program or decoder state")
    program.load_state_dict(program_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    program.eval()
    decoder.eval()
    return program, decoder


def _route_queries(
    parent,
    *,
    operation: str,
    span: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation=operation,
    )
    state = parent.initial_state(batch.batch_size, device=batch.input_frames.device)
    zeros = torch.zeros(batch.batch_size, device=batch.input_frames.device)
    feedback = ControllerFeedback(
        torch.zeros(batch.batch_size, ACTION_WIDTH, device=batch.input_frames.device),
        zeros,
        torch.ones(batch.batch_size, device=batch.input_frames.device),
        zeros,
    )
    query_events: list[torch.Tensor] = []
    with torch.no_grad():
        for frame in batch.input_frames.transpose(0, 1):
            _, state = parent.step_streams({"vision": frame}, state, feedback)
        for frame in batch.distractor_frames.transpose(0, 1):
            _, state = parent.step_streams({"vision": frame}, state, feedback)
        for frame in batch.query_frames.transpose(0, 1):
            query_events.append(parent.encoders["vision"](frame))
            _, state = parent.step_streams({"vision": frame}, state, feedback)
    occupancy = state.event_window.present.to(state.hidden).float()
    event_summary = torch.stack(query_events, dim=1).mean(dim=1)
    return F.normalize(
        torch.cat([event_summary, occupancy * 8.0], dim=-1),
        dim=-1,
    )


def _train_router(
    parent,
    candidate_keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> dict[str, object]:
    router = OpaqueAddressRouter(
        width=int(candidate_keys.shape[-1]),
        hidden=64,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_count = int(candidate_keys.shape[0])
    router.train()
    for update in range(updates):
        family = update % len(PROGRAMS)
        _, operation, span = PROGRAMS[family]
        queries = _route_queries(
            parent,
            operation=operation,
            span=span,
            count=batch_size,
            seed=seed + update * 10_007,
        )
        competitor = torch.randint(
            row_count - 1,
            (batch_size,),
            generator=generator,
        )
        competitor = competitor + (competitor >= family).to(torch.long)
        attempted = torch.stack(
            [
                torch.full((batch_size,), family, dtype=torch.long),
                competitor,
            ],
            dim=1,
        )
        outcomes = (attempted == family).to(torch.float32)
        if shuffle_outcomes:
            shuffled_signs = torch.zeros(batch_size, dtype=torch.long)
            shuffled_signs[: batch_size // 2] = 1
            shuffled_signs = shuffled_signs[
                torch.randperm(batch_size, generator=generator)
            ]
            outcomes = torch.stack(
                [shuffled_signs, 1 - shuffled_signs],
                dim=1,
            ).to(torch.float32)
        loss, _ = paired_counterfactual_ranking_loss(
            router(queries, candidate_keys),
            attempted,
            outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
    router.eval()
    return {
        "router": router,
        "accounting": {
            "unique_route_lifetimes": updates * batch_size,
            "unique_route_verifier_bits": updates * batch_size * 2,
            "route_optimizer_updates": updates,
            "replayed_route_examples": 0,
        },
    }


def _rollout_capability(
    parent,
    program: ExternalCapabilityProgram,
    decoder: OpaqueProtocolDecoder,
    batch,
    *,
    train: bool,
) -> dict[str, torch.Tensor]:
    device = batch.input_frames.device
    state = parent.initial_state(batch.batch_size, device=device)
    capability_state = program.initial_state(batch.batch_size, device=device)
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
        nonlocal state, capability_state
        with torch.no_grad():
            event = encoder(frame)
            output, state = parent.step_streams({"vision": frame}, state, feedback)
        adapted, capability_state = program.step(
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
            action = torch.multinomial(
                probabilities * 0.9 + 0.05,
                1,
            ).squeeze(1)
        else:
            action = logits.argmax(dim=-1)
        reward = (action == correct).to(logits.dtype)
        selected = logits.gather(1, action.unsqueeze(1)).squeeze(1)
        losses.append(F.binary_cross_entropy_with_logits(selected, reward))
        rewards.append(reward)
        previous_action = F.one_hot(action, ACTION_WIDTH).to(logits.dtype)
        previous_reward = reward
        previous_propensity = probabilities.gather(
            1,
            action.unsqueeze(1),
        ).squeeze(1).detach()
        previous_has_feedback = torch.ones_like(previous_reward)
    return {"loss": torch.stack(losses).mean(), "rewards": torch.stack(rewards, dim=1)}


def _train_capability(
    parent,
    program: ExternalCapabilityProgram,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    span: int,
    updates: int,
    batch_size: int,
    seed: int,
    audit_count: int,
    eval_every: int,
    learning_rate: float,
    generated_composition_ids: tuple[int, ...] | None = None,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    trainable = list(program.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    program.train()
    decoder.train()
    for update in range(1, updates + 1):
        target = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation=operation,
            generated_composition_ids=generated_composition_ids,
        )
        target_result = _rollout_capability(
            parent,
            program,
            decoder,
            target,
            train=True,
        )
        auxiliary = generate_sequence_memory_batch(
            batch_size,
            span=2,
            distractors=1,
            seed=seed + 5_000_003 + update * 20_021,
            operation="forward",
        )
        auxiliary_result = _rollout_capability(
            parent,
            program,
            decoder,
            auxiliary,
            train=True,
        )
        loss = target_result["loss"] + auxiliary_result["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "unique_logical_lifetimes": update * batch_size * 2,
                "unique_verifier_bits": update * batch_size * (span + 2),
                "training_accuracy": float(target_result["rewards"].mean()),
                "loss": float(loss.detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            program.eval()
            decoder.eval()
            heldout = generate_sequence_memory_batch(
                audit_count,
                span=span,
                distractors=1,
                seed=seed + 1_000_000 + update,
                operation=operation,
                generated_composition_ids=generated_composition_ids,
            )
            progress.append(
                {
                    "update": update,
                    "unique_verifier_bits": update * batch_size * span,
                    "heldout_accuracy": float(
                        _rollout_capability(
                            parent,
                            program,
                            decoder,
                            heldout,
                            train=False,
                        )["rewards"].mean()
                    ),
                }
            )
            program.train()
            decoder.train()
    program.eval()
    decoder.eval()
    return history, progress


@torch.no_grad()
def _capability_accuracy(
    parent,
    program: ExternalCapabilityProgram,
    decoder: OpaqueProtocolDecoder,
    *,
    operation: str,
    span: int,
    count: int,
    seed: int,
    generated_composition_ids: tuple[int, ...] | None = None,
) -> float:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation=operation,
        generated_composition_ids=generated_composition_ids,
    )
    return float(
        _rollout_capability(parent, program, decoder, batch, train=False)[
            "rewards"
        ].mean()
    )


def _test_queries(parent, *, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.cat(
        [
            _route_queries(
                parent,
                operation=operation,
                span=span,
                count=count,
                seed=seed + index * 10_007,
            )
            for index, (_label, operation, span) in enumerate(PROGRAMS)
        ]
    )
    targets = torch.arange(len(PROGRAMS)).repeat_interleave(count)
    return queries, targets


@torch.no_grad()
def _route_accuracy(router, queries, targets, candidate_keys) -> float:
    return float(
        (router(queries, candidate_keys).argmax(dim=-1) == targets)
        .float()
        .mean()
    )


@torch.no_grad()
def _permuted_accuracy(router, queries, targets, candidate_keys) -> float:
    permutation = torch.arange(candidate_keys.shape[0] - 1, -1, -1)
    predictions = router(queries, candidate_keys[permutation]).argmax(dim=-1)
    return float((permutation[predictions] == targets).float().mean())


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if tuple(args.stages) != tuple(program[2] for program in PROGRAMS):
        raise ValueError("this audit requires the configured span-4 procedures")
    if min(
        args.parent_updates,
        args.updates,
        args.route_updates,
        args.batch_size,
        args.route_batch_size,
        args.audit_count,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.route_batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch sizes and audit count must be even")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    parent_history, parent_progress = _train_with_progress(
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
    parent_behavior = _accuracy(
        parent,
        operation="forward",
        span=2,
        count=args.audit_count,
        seed=args.seed + 40_002,
    )
    parent_digest_before = _digest_core(parent, ())
    parent_stable_bits = _stable_bits(
        parent_progress,
        threshold=0.75,
        bits_per_update=args.batch_size * 2,
    )

    bank_path = args.report_out.parent / "artifact_bank"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    bank = ExecutableArtifactMemory(
        bank_path,
        width=48,
        capacity=len(PROGRAMS),
        write_match_threshold=0.99999,
    )
    histories: dict[str, object] = {
        "parent": {"history": parent_history, "progress": parent_progress}
    }
    stable_bits: dict[str, int | None] = {}
    for index, (label, operation, span) in enumerate(PROGRAMS):
        program, decoder = _new_capability(args.seed + index + 1)
        history, progress = _train_capability(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed + 200 * (index + 1),
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
        )
        route_key = F.normalize(
            _route_queries(
                parent,
                operation=operation,
                span=span,
                count=args.audit_count,
                seed=args.seed + 50_000 + index,
            ).mean(dim=0),
            dim=0,
        )
        bank.put(route_key, _artifact(program, decoder))
        stable_bits[label] = _stable_bits(
            progress,
            threshold=0.75,
            bits_per_update=args.batch_size * span,
        )
        histories[label] = {
            "history": history,
            "progress": progress,
            "stable_bits_to_threshold": stable_bits[label],
        }
    bank = ExecutableArtifactMemory.load(bank_path)

    candidates = bank.address_rows()
    candidate_keys = torch.stack([key for _, key in candidates])
    router_result = _train_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
        shuffle_outcomes=False,
    )
    shuffled_result = _train_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=True,
    )
    router = router_result["router"]
    shuffled_router = shuffled_result["router"]
    test_queries, test_targets = _test_queries(
        parent,
        count=args.audit_count,
        seed=args.seed + 80_000,
    )
    route_accuracy = _route_accuracy(router, test_queries, test_targets, candidate_keys)
    shuffled_route_accuracy = _route_accuracy(
        shuffled_router,
        test_queries,
        test_targets,
        candidate_keys,
    )
    permuted_accuracy = _permuted_accuracy(
        router,
        test_queries,
        test_targets,
        candidate_keys,
    )

    selected_behavior: dict[str, float] = {}
    selected_parent_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    selected_rows: dict[str, int] = {}
    for family, (label, operation, span) in enumerate(PROGRAMS):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        row = int(torch.mode(router(family_queries, candidate_keys).argmax(dim=-1)).values)
        selected_rows[label] = row
        _, artifact = bank.promote_index(row)
        program, decoder = _load_artifact(artifact)
        selected_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 100_000 + family,
        )
        selected_parent_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation="forward",
            span=2,
            count=args.audit_count,
            seed=args.seed + 140_000 + family,
        )
        wrong_row = (row + 1) % len(PROGRAMS)
        _, wrong_artifact = bank.promote_index(wrong_row)
        wrong_program, wrong_decoder = _load_artifact(wrong_artifact)
        wrong_behavior[label] = _capability_accuracy(
            parent,
            wrong_program,
            wrong_decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 100_000 + family,
        )

    router_path = args.report_out.parent / "router.pt"
    route_state = PersistentOpaqueStateStore(
        router_path,
        configuration={
            "component": "parent-conditioned-artifact-route",
            "schema": "neural-computer.opaque-address-router.v1",
            "width": 48,
            "hidden": 64,
            "candidate_count": len(PROGRAMS),
        },
    )
    route_state.save_module(router)
    reloaded = ExecutableArtifactMemory.load(bank_path)
    reloaded_candidates = reloaded.address_rows()
    reloaded_keys = torch.stack([key for _, key in reloaded_candidates])
    reloaded_router = OpaqueAddressRouter(width=48, hidden=64)
    route_state.load_module(reloaded_router)
    reloaded_router.eval()
    reloaded_route_accuracy = _route_accuracy(
        reloaded_router,
        test_queries,
        test_targets,
        reloaded_keys,
    )
    reloaded_candidate_exact = all(
        torch.equal(reloaded_keys[index], candidate_keys[index])
        for index in range(len(candidate_keys))
    )
    reloaded_behavior: dict[str, float] = {}
    reloaded_parent_behavior: dict[str, float] = {}
    reloaded_rows: dict[str, int] = {}
    for family, (label, operation, span) in enumerate(PROGRAMS):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        row = int(
            torch.mode(
                reloaded_router(family_queries, reloaded_keys).argmax(dim=-1)
            ).values
        )
        reloaded_rows[label] = row
        _, artifact = reloaded.promote_index(row)
        program, decoder = _load_artifact(artifact)
        reloaded_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 100_000 + family,
        )
        reloaded_parent_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation="forward",
            span=2,
            count=args.audit_count,
            seed=args.seed + 140_000 + family,
        )

    artifact_name = reloaded.paths[0]
    if artifact_name is None:
        raise RuntimeError("reloaded bank has no artifact path")
    artifact_path = bank_path / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(bank_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)

    parent_digest_after = _digest_core(parent, ())
    router_accounting = router_result["accounting"]
    shuffled_accounting = shuffled_result["accounting"]
    report = {
        "schema": "neural-computer.parent-conditioned-artifact-bank-report.v3",
        "claim_boundary": (
            "Three independently learned external recurrent capability programs "
            "are stored, routed, and reloaded through one opaque artifact bank "
            "while the shared controller remains frozen. Output decoding is "
            "capability-local and replaceable. This is not general continual "
            "learning or unrestricted program induction."
        ),
        "seed": args.seed,
        "programs": [
            {"label": label, "operation": operation, "span": span}
            for label, operation, span in PROGRAMS
        ],
        "parent_updates": args.parent_updates,
        "updates_per_artifact": args.updates,
        "batch_size": args.batch_size,
        "route_updates": args.route_updates,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "external_program": ExternalCapabilityProgram(
            EVENT_WIDTH,
            ACTION_WIDTH,
            INTENTION_WIDTH,
            context_hidden=CAPABILITY_CONTEXT_HIDDEN,
            context_width=CAPABILITY_CONTEXT_WIDTH,
            adapter_hidden=CAPABILITY_ADAPTER_HIDDEN,
        ).configuration(),
        "parent_behavior": parent_behavior,
        "parent_stable_bits_to_threshold": parent_stable_bits,
        "selected_behavior": selected_behavior,
        "stable_bits_to_threshold": stable_bits,
        "selected_parent_behavior": selected_parent_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_behavior": reloaded_behavior,
        "reloaded_parent_behavior": reloaded_parent_behavior,
        "route_accuracy": route_accuracy,
        "reward_shuffled_route_accuracy": shuffled_route_accuracy,
        "candidate_permutation_accuracy": permuted_accuracy,
        "reloaded_route_accuracy": reloaded_route_accuracy,
        "selected_rows": selected_rows,
        "reloaded_rows": reloaded_rows,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "histories": histories,
        "accounting": {
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.updates * args.batch_size * len(PROGRAMS) * 2
            ),
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.updates
                * args.batch_size
                * sum(span + 2 for _, _, span in PROGRAMS)
            ),
            "optimizer_updates": args.parent_updates + args.updates * len(PROGRAMS),
            "route_optimizer_updates": args.route_updates * 2,
            "route_unique_lifetimes": (
                router_accounting["unique_route_lifetimes"]
                + shuffled_accounting["unique_route_lifetimes"]
            ),
            "route_unique_verifier_bits": (
                router_accounting["unique_route_verifier_bits"]
                + shuffled_accounting["unique_route_verifier_bits"]
            ),
            "replayed_examples": 0,
            "route_replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "three_artifacts_present": len(bank.occupied) == 3,
            "parent_stable": parent_stable_bits is not None,
            "learned_route_at_least_90": route_accuracy >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route_accuracy <= 0.50,
            "candidate_permutation_invariant": permuted_accuracy >= 0.90,
            "parent_retained": all(
                selected_parent_behavior[label] >= parent_behavior - 0.02
                for label, _, _ in PROGRAMS
            ),
            "all_selected_programs_mastered": all(
                selected_behavior[label] >= 0.75
                and stable_bits[label] is not None
                for label, _, _ in PROGRAMS
            ),
            "wrong_program_is_causal": all(
                selected_behavior[label] > wrong_behavior[label] + 0.05
                for label, _, _ in PROGRAMS
            ),
            "reloaded_route_preserved": reloaded_route_accuracy >= 0.90,
            "reloaded_rows_match": reloaded_rows == selected_rows,
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "reloaded_behavior_preserved": all(
                reloaded_behavior[label] >= selected_behavior[label] - 0.05
                for label, _, _ in PROGRAMS
            ),
            "reloaded_parent_retained": all(
                reloaded_parent_behavior[label] >= parent_behavior - 0.02
                for label, _, _ in PROGRAMS
            ),
            "core_unchanged": parent_digest_before == parent_digest_after,
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument(
        "--stages",
        type=int,
        nargs=3,
        default=[program[2] for program in PROGRAMS],
    )
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-updates", type=int, default=512)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "route_accuracy": report["route_accuracy"],
                "reward_shuffled_route_accuracy": report[
                    "reward_shuffled_route_accuracy"
                ],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "selected_behavior": report["selected_behavior"],
                "wrong_behavior": report["wrong_behavior"],
                "reloaded_route_accuracy": report["reloaded_route_accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
