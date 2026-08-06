"""Qualify learned opaque-row eviction with a frozen controller.

The controller is acquired once, then frozen. A separately versioned memory
policy ranks candidate rows using only controller-native write context and
opaque row tensors. Paired common-random arms force different physical rows
and train the ranking from scalar recall outcomes.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from experiments.memory_retention_amodal.environment import (
    OutcomeOnlyRetentionVerifier,
)
from experiments.memory_retention_amodal.train import (
    _build_external_write_policy,
    _empty,
    _event,
    _feedback,
    _probe_without_writing,
    _retention_slots,
    build_runtime,
    seed_everything,
    train_curriculum,
)
from neural_computer import (
    AmodalControllerRuntime,
    ExternalMemoryEvictionPolicy,
    MemoryEvictionObservation,
    MemoryWriteObservation,
    PersistentContentAddressedMemory,
    paired_counterfactual_ranking_loss,
)


def _build_eviction_policy(
    runtime: AmodalControllerRuntime,
) -> ExternalMemoryEvictionPolicy:
    width = runtime.controller.width
    return ExternalMemoryEvictionPolicy(
        event_width=runtime.event_width,
        hidden_width=width,
        workspace_width=width,
        key_width=width,
        value_width=width,
        memory_read_width=width,
        action_width=2,
        controller_write_context_width=width * 4,
        controller_write_relevance_width=1,
        candidate_key_width=width,
        candidate_value_width=width,
    )


def _controller_write_observation(
    runtime: AmodalControllerRuntime,
    state: Any,
    action: torch.Tensor,
    reward: torch.Tensor,
    propensity: torch.Tensor,
    scope: torch.Tensor,
) -> tuple[Any, MemoryWriteObservation, torch.Tensor]:
    previous_event = state.latest_event.detach()
    batch = action.shape[0]
    opaque_action = torch.nn.functional.one_hot(action, num_classes=2).to(
        torch.float32
    )
    output, state = runtime.step_events(
        _empty(batch, runtime.event_width),
        state,
        _feedback(
            batch,
            action=opaque_action,
            reward=reward,
            propensity=propensity,
            has_feedback=torch.ones(batch),
        ),
        memory_scope=scope,
        memory_write_override=torch.zeros(batch),
        memory_write_gradient=False,
    )
    controller_output = output.controller
    memory_read_value = (
        torch.zeros_like(controller_output.memory_value)
        if controller_output.memory_read is None
        else controller_output.memory_read.value
    )
    memory_read_hit = (
        torch.zeros(batch)
        if controller_output.memory_read is None
        else controller_output.memory_read.hit.to(torch.float32)
    )
    if controller_output.memory_write_context is None:
        raise RuntimeError("controller did not emit a memory write context")
    if controller_output.memory_write_relevance is None:
        raise RuntimeError("controller did not emit memory write relevance")
    observation = MemoryWriteObservation(
        event=previous_event,
        hidden=state.hidden.detach(),
        workspace_read=controller_output.workspace_read.detach(),
        query_key=controller_output.memory_query_key.detach(),
        write_value=controller_output.memory_value.detach(),
        controller_write_proposal=controller_output.memory_write_strength.detach(),
        controller_write_context=controller_output.memory_write_context.detach(),
        controller_write_relevance=controller_output.memory_write_relevance.detach(),
        memory_read_value=memory_read_value.detach(),
        memory_read_hit=memory_read_hit.detach(),
        action=opaque_action.detach(),
        reward=reward.detach(),
        propensity=propensity.detach(),
        has_feedback=torch.ones(batch),
    )
    return state, observation, controller_output.memory_key.detach()


def _candidate_scores(
    policy: ExternalMemoryEvictionPolicy,
    observation: MemoryWriteObservation,
    candidates: Any,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for index in range(candidates.keys.shape[1]):
        scores.append(
            policy(
                MemoryEvictionObservation(
                    write=observation,
                    candidate_key=candidates.keys[:, index],
                    candidate_value=candidates.values[:, index],
                    candidate_strength=candidates.strengths[:, index],
                    candidate_timestamp=candidates.timestamps[:, index],
                    candidate_occupied=candidates.occupied[:, index],
                )
            )
        )
    return torch.stack(scores, dim=1)


def _store_eviction(
    runtime: AmodalControllerRuntime,
    state: Any,
    action: torch.Tensor,
    reward: torch.Tensor,
    propensity: torch.Tensor,
    scope: torch.Tensor,
    *,
    policy: ExternalMemoryEvictionPolicy | None,
    forced_index: torch.Tensor | None = None,
    greedy: bool = True,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    state, observation, memory_key = _controller_write_observation(
        runtime, state, action, reward, propensity, scope
    )
    candidates = runtime.memory.candidates(scope)
    scores = (
        _candidate_scores(policy, observation, candidates)
        if policy is not None
        else torch.zeros_like(candidates.strengths)
    )
    occupied = candidates.occupied
    full = occupied.all(dim=1)
    selected = torch.full(
        (action.shape[0],),
        -1,
        dtype=torch.long,
        device=action.device,
    )
    log_probability = torch.zeros(action.shape[0])
    if bool(full.any()):
        if policy is None:
            chosen = torch.argmin(candidates.strengths, dim=1)
        elif forced_index is not None:
            chosen = forced_index.reshape(-1).to(dtype=torch.long)
        elif greedy:
            chosen = scores.masked_fill(~occupied, -torch.inf).argmax(dim=1)
        else:
            distribution = Categorical(logits=scores.masked_fill(~occupied, -torch.inf))
            chosen = distribution.sample()
        selected = torch.where(full, chosen, selected)
        if policy is not None:
            log_scores = torch.log_softmax(
                scores.masked_fill(~occupied, -torch.inf), dim=1
            )
            log_probability = torch.where(
                full,
                log_scores.gather(1, chosen[:, None]).squeeze(1),
                log_probability,
            )
    receipt = runtime.memory.write(
        memory_key,
        observation.write_value,
        torch.ones(action.shape[0]),
        timestamp=torch.zeros(action.shape[0]),
        scope=scope,
        target_index=selected,
    )
    return state, scores, log_probability, receipt.indices


def _episode_order(
    verifier: OutcomeOnlyRetentionVerifier, order: str
) -> torch.Tensor:
    if order == "random":
        return verifier.order
    if order == "target_first":
        distractors = verifier.order[verifier.order != verifier.query_slot[:, None]].reshape(
            verifier.batch_size, verifier.slot_count - 1
        )
        return torch.cat((verifier.query_slot[:, None], distractors), dim=1)
    if order == "target_last":
        distractors = verifier.order[verifier.order != verifier.query_slot[:, None]].reshape(
            verifier.batch_size, verifier.slot_count - 1
        )
        return torch.cat((distractors, verifier.query_slot[:, None]), dim=1)
    if order == "target_middle":
        if verifier.slot_count != 3:
            raise ValueError("target_middle requires exactly three slots")
        distractors = verifier.order[verifier.order != verifier.query_slot[:, None]].reshape(
            verifier.batch_size, verifier.slot_count - 1
        )
        return torch.stack(
            (distractors[:, 0], verifier.query_slot, distractors[:, 1]), dim=1
        )
    if order == "balanced":
        return _retention_slots(verifier, "balanced")
    raise ValueError(f"unknown eviction order: {order}")


def _run_episode(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    policy: ExternalMemoryEvictionPolicy | None,
    *,
    order: str,
    clear: bool = False,
    corrupt: bool = False,
    random_eviction: bool = False,
    reward_shuffle: bool = False,
    forced_eviction: torch.Tensor | None = None,
    shared_recall_uniform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verifier.reset()
    batch = verifier.batch_size
    scope = torch.arange(batch, dtype=torch.long)
    runtime.memory.clear()
    state = runtime.initial_state(batch, device="cpu")
    _, state = runtime.controller.step(
        _event(tokens[verifier.query_slot]), state, _feedback(batch), memory=None
    )
    slots = _episode_order(verifier, order)
    eviction_scores = torch.zeros(batch, runtime.memory.capacity)
    eviction_log_probability = torch.zeros(batch)
    for position in range(verifier.slot_count):
        slot = slots[:, position]
        action, propensity, _, state = _probe_without_writing(
            runtime, state, _event(tokens[slot])
        )
        reward = verifier.score_probe(slot, action)
        if reward_shuffle:
            reward = torch.randint(0, 2, reward.shape).to(torch.float32)
        if position == verifier.slot_count - 1 and random_eviction:
            forced_eviction = torch.randint(0, runtime.memory.capacity, (batch,))
        state, scores, log_probability, _ = _store_eviction(
            runtime,
            state,
            action,
            reward,
            propensity,
            scope,
            policy=policy,
            forced_index=forced_eviction if position == verifier.slot_count - 1 else None,
            greedy=not random_eviction,
        )
        if position == verifier.slot_count - 1:
            eviction_scores = scores
            eviction_log_probability = log_probability
    if clear:
        runtime.memory.clear()
    elif corrupt:
        runtime.memory.values.zero_()
    query_output, _ = runtime.step_events(
        _event(tokens[verifier.query_slot]),
        runtime.initial_state(batch, device="cpu"),
        _feedback(batch),
        memory_scope=scope,
        memory_write_override=torch.zeros(batch),
        memory_write_gradient=False,
    )
    distribution = Categorical(logits=query_output.decoded["protocol"])
    if shared_recall_uniform is None:
        action = distribution.sample()
    else:
        uniform = shared_recall_uniform.reshape(-1).to(torch.float32)
        action = (
            uniform[:, None] >= distribution.probs.cumsum(dim=-1)
        ).sum(dim=-1).clamp_max(distribution.probs.shape[-1] - 1)
    return (
        verifier.score_recall(action),
        eviction_scores,
        eviction_log_probability,
    )


def _train_eviction_step(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    policy: ExternalMemoryEvictionPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    reward_shuffle: bool,
) -> tuple[float, float]:
    paired = verifier.duplicate_rows(2)
    batch = verifier.batch_size
    forced = torch.arange(batch * 2, dtype=torch.long) % 2
    shared_uniform = torch.rand(batch).repeat_interleave(2)
    losses: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    attempted = torch.tensor([[0, 1]], dtype=torch.long).repeat(batch, 1)
    for factor_order in ("target_first", "target_middle"):
        recall, scores, _ = _run_episode(
            runtime,
            paired,
            tokens,
            policy,
            order=factor_order,
            reward_shuffle=reward_shuffle,
            forced_eviction=forced,
            shared_recall_uniform=shared_uniform,
        )
        utilities = recall.reshape(batch, 2)
        loss, advantage = paired_counterfactual_ranking_loss(
            scores[::2], attempted, utilities
        )
        losses.append(loss)
        advantages.append(advantage)
    loss = torch.stack(losses).mean()
    advantage = torch.stack(advantages).mean(dim=0)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in policy.parameters() if parameter.grad is not None]
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
    optimizer.step()
    return float(loss.detach()), float(advantage.mean())


@torch.no_grad()
def _evaluate(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    policy: ExternalMemoryEvictionPolicy | None,
    *,
    order: str,
    episodes: int = 64,
    clear: bool = False,
    corrupt: bool = False,
    random_eviction: bool = False,
) -> float:
    total = 0.0
    for _ in range(episodes):
        recall, _, _ = _run_episode(
            runtime,
            verifier,
            tokens,
            policy,
            order=order,
            clear=clear,
            corrupt=corrupt,
            random_eviction=random_eviction,
        )
        total += float(recall.sum())
    return total / (episodes * verifier.batch_size)


@torch.no_grad()
def _evaluate_persistent(
    runtime: AmodalControllerRuntime,
    verifier: OutcomeOnlyRetentionVerifier,
    tokens: torch.Tensor,
    policy: ExternalMemoryEvictionPolicy,
    *,
    episodes: int = 32,
) -> dict[str, float | bool]:
    original_memory = runtime.memory
    if original_memory is None:
        raise RuntimeError("persistent eviction audit requires a memory backend")
    scope = torch.arange(verifier.batch_size, dtype=torch.long)
    total = 0.0
    recovery_total = 0.0
    corruption_rejected = False

    def query() -> float:
        output, _ = runtime.step_events(
            _event(tokens[verifier.query_slot]),
            runtime.initial_state(verifier.batch_size, device="cpu"),
            _feedback(verifier.batch_size),
            memory_scope=scope,
            memory_write_override=torch.zeros(verifier.batch_size),
            memory_write_gradient=False,
        )
        action = Categorical(logits=output.decoded["protocol"]).sample()
        return float(verifier.score_recall(action).sum())

    with tempfile.TemporaryDirectory(prefix="neural-computer-eviction-") as directory:
        path = Path(directory) / "eviction-memory.pt"
        persistent = PersistentContentAddressedMemory(
            width=original_memory.width,
            capacity=original_memory.capacity,
            path=path,
            write_threshold=original_memory.write_threshold,
            query_temperature=original_memory.query_temperature,
            write_match_threshold=original_memory.write_match_threshold,
            scope_capacity=original_memory.scope_capacity,
        )
        runtime.memory = persistent
        try:
            for _ in range(episodes):
                _run_episode(
                    runtime,
                    verifier,
                    tokens,
                    policy,
                    order="target_first",
                )
                reloaded = PersistentContentAddressedMemory(
                    width=original_memory.width,
                    capacity=original_memory.capacity,
                    path=path,
                    write_threshold=original_memory.write_threshold,
                    query_temperature=original_memory.query_temperature,
                    write_match_threshold=original_memory.write_match_threshold,
                    scope_capacity=original_memory.scope_capacity,
                )
                runtime.memory = reloaded
                total += query()
                runtime.memory = persistent

            payload = torch.load(path, map_location="cpu", weights_only=False)
            corrupted = dict(payload)
            corrupted_state = dict(payload["state_dict"])
            corrupted_values = corrupted_state["values"].clone()
            corrupted_values.reshape(-1)[0] += 1.0
            corrupted_state["values"] = corrupted_values
            corrupted["state_dict"] = corrupted_state
            torch.save(corrupted, path)
            try:
                PersistentContentAddressedMemory(
                    width=original_memory.width,
                    capacity=original_memory.capacity,
                    path=path,
                    write_threshold=original_memory.write_threshold,
                    query_temperature=original_memory.query_temperature,
                    write_match_threshold=original_memory.write_match_threshold,
                    scope_capacity=original_memory.scope_capacity,
                )
            except ValueError as error:
                corruption_rejected = "checksum" in str(error)
            persistent.snapshot(path)
            recovered = PersistentContentAddressedMemory(
                width=original_memory.width,
                capacity=original_memory.capacity,
                path=path,
                write_threshold=original_memory.write_threshold,
                query_temperature=original_memory.query_temperature,
                write_match_threshold=original_memory.write_match_threshold,
                scope_capacity=original_memory.scope_capacity,
            )
            runtime.memory = recovered
            recovery_total = query()
        finally:
            runtime.memory = original_memory
    return {
        "reload_intact_recall": total / (episodes * verifier.batch_size),
        "corruption_rejected": corruption_rejected,
        "recovery_intact_recall": recovery_total / verifier.batch_size,
    }


def run_experiment(
    *,
    parent_steps: int,
    eviction_steps: int,
    seed: int,
    batch_size: int = 16,
    reward_shuffle: bool = False,
    report_out: Path | None = None,
) -> dict[str, Any]:
    if min(parent_steps, eviction_steps) < 1:
        raise ValueError("training steps must be positive")
    seed_everything(seed)
    runtime = build_runtime(
        seed=seed,
        batch_size=batch_size,
        memory_capacity=2,
        event_window_capacity=4,
        memory_scope_capacity=batch_size * 2,
    )
    tokens = torch.randn(3, runtime.event_width)
    parent_verifier = OutcomeOnlyRetentionVerifier(
        batch_size=batch_size, seed=seed + 10, slot_count=3
    )
    parent_writer = _build_external_write_policy(runtime)
    _, parent_accounting = train_curriculum(
        runtime,
        parent_verifier,
        tokens,
        phase1_steps=parent_steps,
        phase2_steps=64,
        seed=seed,
        reward_shuffle=reward_shuffle,
        retention_order="balanced",
        write_credit="external_overwrite_v2",
        parent_credit="counterfactual_three_factor",
        external_write_policy=parent_writer,
        randomize_event_tokens=True,
        retention_token_reuse_steps=4,
    )
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    runtime.eval()
    policy = _build_eviction_policy(runtime)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
    verifier = OutcomeOnlyRetentionVerifier(
        batch_size=batch_size, seed=seed + 20, slot_count=3
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, eviction_steps + 1):
        episode_tokens = torch.randn_like(tokens)
        loss, advantage = _train_eviction_step(
            runtime,
            verifier,
            episode_tokens,
            policy,
            optimizer,
            reward_shuffle=reward_shuffle,
        )
        if step == 1 or step % 16 == 0 or step == eviction_steps:
            history.append({"step": step, "loss": loss, "mean_advantage": advantage})
    elapsed = time.perf_counter() - started
    eval_verifier = OutcomeOnlyRetentionVerifier(
        batch_size=batch_size, seed=seed + 100, slot_count=3
    )
    eval_tokens = torch.randn_like(tokens)
    conditions = {
        "learned_balanced": _evaluate(
            runtime, eval_verifier, eval_tokens, policy, order="balanced"
        ),
        "learned_target_first": _evaluate(
            runtime, eval_verifier, eval_tokens, policy, order="target_first"
        ),
        "learned_target_last": _evaluate(
            runtime, eval_verifier, eval_tokens, policy, order="target_last"
        ),
        "clear_memory": _evaluate(
            runtime, eval_verifier, eval_tokens, policy, order="balanced", clear=True
        ),
        "corrupt_memory": _evaluate(
            runtime, eval_verifier, eval_tokens, policy, order="balanced", corrupt=True
        ),
        "random_eviction": _evaluate(
            runtime,
            eval_verifier,
            eval_tokens,
            policy,
            order="balanced",
            random_eviction=True,
        ),
        "random_target_first": _evaluate(
            runtime,
            eval_verifier,
            eval_tokens,
            policy,
            order="target_first",
            random_eviction=True,
        ),
        "random_target_last": _evaluate(
            runtime,
            eval_verifier,
            eval_tokens,
            policy,
            order="target_last",
            random_eviction=True,
        ),
        "strength_eviction": _evaluate(
            runtime, eval_verifier, eval_tokens, None, order="balanced"
        ),
        "strength_target_first": _evaluate(
            runtime, eval_verifier, eval_tokens, None, order="target_first"
        ),
        "strength_target_last": _evaluate(
            runtime, eval_verifier, eval_tokens, None, order="target_last"
        ),
    }
    persistent = (
        _evaluate_persistent(
            runtime,
            OutcomeOnlyRetentionVerifier(
                batch_size=batch_size, seed=seed + 200, slot_count=3
            ),
            eval_tokens,
            policy,
        )
        if not reward_shuffle
        else {
            "audit_enabled": False,
            "reload_intact_recall": None,
            "corruption_rejected": None,
            "recovery_intact_recall": None,
        }
    )
    accounting = {
        "parent_steps_requested": parent_steps,
        "parent_updates": int(parent_accounting["phase1_updates"]),
        "eviction_updates": eviction_steps,
        "optimizer_updates": eviction_steps,
        "counterfactual_eviction_factors": 2,
        "replayed_examples": 0,
        "unique_logical_lifetimes": eviction_steps * batch_size * 3 * 2 * 2,
        "unique_verifier_bits": eviction_steps * batch_size * 3 * 2 * 2,
        "wall_time_seconds": elapsed,
        "mean_inference_latency_ms": elapsed / max(1, eviction_steps * batch_size) * 1000,
        "parent_stable": bool(parent_accounting["parent_stable"]),
    }
    promotion = (
        not reward_shuffle
        and accounting["parent_stable"]
        and conditions["learned_target_first"] >= 0.75
        and conditions["learned_target_last"] >= 0.75
        and conditions["learned_balanced"] >= 0.75
        and conditions["clear_memory"] <= 0.60
        and conditions["corrupt_memory"] <= 0.60
        and conditions["random_target_first"]
        < conditions["learned_target_first"] - 0.15
        and conditions["strength_target_first"] < conditions["learned_target_first"] - 0.10
        and float(persistent["reload_intact_recall"] or 0.0) >= 0.80
        and bool(persistent["corruption_rejected"])
        and float(persistent["recovery_intact_recall"] or 0.0) >= 0.80
    )
    report: dict[str, Any] = {
        "schema": "neural-computer.learned-eviction-experiment.v1",
        "experiment": "outcome-only-learned-eviction",
        "seed": seed,
        "parent_steps": parent_steps,
        "eviction_steps": eviction_steps,
        "batch_size": batch_size,
        "slot_count": 3,
        "memory_capacity": 2,
        "reward_shuffle_control": reward_shuffle,
        "learner_visible_inputs": [
            "opaque target cue event",
            "opaque slot event",
            "opaque attempted action",
            "scalar probe outcome",
            "opaque candidate key/value/strength/timestamp",
            "scalar recall outcome",
        ],
        "parent": parent_accounting,
        "parent_training": {
            "randomize_event_tokens": True,
            "retention_token_reuse_steps": 4,
            "controller_frozen_before_eviction_training": True,
        },
        "eviction_training": {
            "fresh_tokens_each_update": True,
            "counterfactual_factors": ["target_first", "target_middle"],
            "row_labels_visible_to_policy": False,
        },
        "policy": policy.configuration(),
        "history": history,
        "conditions": conditions,
        "persistent_memory": persistent,
        "accounting": accounting,
        "promotion_gate": {
            "learned_order_min": 0.75,
            "clear_max": 0.60,
            "corrupt_max": 0.60,
            "random_target_first_gap_min": 0.15,
            "strength_target_first_gap_min": 0.10,
            "parent_stable_required": True,
        },
        "promoted": promotion,
        "claim_boundary": (
            "Narrow learned utility-based eviction for a frozen controller; "
            "no general episodic memory or arbitrary computation claim."
        ),
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-steps", type=int, default=704)
    parser.add_argument("--eviction-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reward-shuffle", action="store_true")
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(
        parent_steps=args.parent_steps,
        eviction_steps=args.eviction_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        reward_shuffle=args.reward_shuffle,
        report_out=args.report_out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
