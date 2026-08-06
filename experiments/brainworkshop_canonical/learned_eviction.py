"""Audit learned outcome-only eviction for the canonical capability bank.

The frozen controller is paired with two unprotected external capabilities.
An independently trainable memory-side policy receives only an opaque incoming
event and scalar capability-history summaries.  It is trained online from
paired fresh verifier outcomes: larger scores mean a capability is more
disposable.  Retention masking remains outside the policy, so learned utility
cannot evict a mastered capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    ContentAddressedMemory,
    ExternalCapabilityEvictionPolicy,
    RetentionPolicyConfig,
    paired_counterfactual_ranking_loss,
)

from .environment import NBackVerifier
from .recursive_capacity_growth import _bank_digest, _context_status
from .runner import CanonicalBrainWorkshopAgent
from .trainer import (
    train_existing_adaptive_relation_capability,
    train_reward_only,
)

BASE_N_BACK = 2
BASE_CUE = 4
CAPABILITY_N_BACKS = (6, 7)
CAPABILITY_CUES = (5, 6)
CANDIDATE_FEATURE_WIDTH = 32
FIXED_CORE_SEED = 2026


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-worlds", type=int, default=8)
    parser.add_argument("--eval-worlds", type=int, default=3)
    parser.add_argument("--old-updates", type=int, default=64)
    parser.add_argument("--candidate-updates", type=int, default=192)
    parser.add_argument("--replacement-updates", type=int, default=512)
    parser.add_argument("--replacement-learning-rate", type=float, default=3e-3)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--policy-steps-per-world", type=int, default=16)
    parser.add_argument("--base-audits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--policy-learning-rate", type=float, default=5e-2)
    parser.add_argument("--policy-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def _retention_config() -> RetentionPolicyConfig:
    # Candidate audits deliberately remain below this threshold so the policy
    # sees unprotected utility histories. The base capability receives enough
    # independent audits to become protected.
    return RetentionPolicyConfig(
        mastery_threshold=0.8,
        min_mastery_observations=64,
        reversal_patience=4,
        recent_window=8,
    )


def _new_agent() -> CanonicalBrainWorkshopAgent:
    return CanonicalBrainWorkshopAgent(
        symbol_count=8,
        n_back=BASE_N_BACK,
        reader_kind="relation",
        retention_config=_retention_config(),
        seed=FIXED_CORE_SEED,
    )


def _rollout_score(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    cue_symbol: int,
    slot: int,
    batch_size: int,
    steps: int,
    seed: int,
    record_retention: bool,
    time_shuffle: bool = False,
    reset_history: bool = False,
) -> tuple[float, int, int]:
    memory = agent.runtime.memory
    if isinstance(memory, ContentAddressedMemory):
        memory.clear()
    verifier = NBackVerifier(
        batch_size=batch_size,
        n_back=n_back,
        steps=steps,
        symbol_count=4,
        cue_symbol=cue_symbol,
        seed=seed,
        time_shuffle=time_shuffle,
    )
    with torch.no_grad():
        rollout = agent.rollout(
            verifier,
            sample=False,
            reset_history=reset_history,
            record_retention=record_retention,
            forced_slot=slot,
        )
    score = float(rollout.eligible_accuracy.mean())
    return score, batch_size * verifier.eligible_trials, batch_size * verifier.steps


def _candidate_feature(
    agent: CanonicalBrainWorkshopAgent,
    slot: int,
) -> torch.Tensor:
    # The policy receives only an opaque learned address. Outcome histories
    # stay in the trainer as scalar utility signals; feeding the direct score
    # into the policy would collapse this audit into a hand-coded threshold.
    return agent.capability_address_for(slot).detach().clone()


def _incoming_context(
    agent: CanonicalBrainWorkshopAgent,
    cue_symbol: int,
) -> torch.Tensor:
    encoder = agent.runtime.encoders["stimulus"]
    return encoder(torch.tensor([cue_symbol], dtype=torch.long))[0].detach()


def _audit_base(
    agent: CanonicalBrainWorkshopAgent,
    *,
    batch_size: int,
    steps: int,
    seed: int,
    audits: int,
) -> dict[str, int | float]:
    scores: list[float] = []
    bits = 0
    outcomes = 0
    for index in range(audits):
        score, run_bits, run_outcomes = _rollout_score(
            agent,
            n_back=BASE_N_BACK,
            cue_symbol=BASE_CUE,
            slot=0,
            batch_size=batch_size,
            steps=steps,
            seed=seed + 100 + index,
            record_retention=True,
        )
        scores.append(score)
        bits += run_bits
        outcomes += run_outcomes
    return {
        "score": min(scores),
        "unique_verifier_bits": bits,
        "verifier_outcome_events": outcomes,
    }


def _build_world(
    *,
    seed: int,
    batch_size: int,
    steps: int,
    old_updates: int,
    candidate_updates: int,
    strong_first: bool,
    incoming_logical_index: int,
    probe_batches: int,
    base_audits: int,
) -> dict[str, Any]:
    agent = _new_agent()
    old_history = train_reward_only(
        agent,
        n_back=BASE_N_BACK,
        updates=old_updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=1e-2,
        cue_symbol=BASE_CUE,
    )
    base_audit = _audit_base(
        agent,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        audits=base_audits,
    )
    candidate_features: list[torch.Tensor] = [_candidate_feature(agent, 0)]
    candidate_history: list[dict[str, Any]] = []
    bits = int(base_audit["unique_verifier_bits"])
    outcomes = int(base_audit["verifier_outcome_events"])
    logical_order = (0, 1) if strong_first else (1, 0)
    candidate_specs: list[tuple[int, int, int]] = []
    for index, logical_index in enumerate(logical_order):
        n_back = CAPABILITY_N_BACKS[logical_index]
        cue = CAPABILITY_CUES[logical_index]
        slot = agent.add_adaptive_relation_capability(
            memory_capacity=7,
            seed=FIXED_CORE_SEED + 5000 + logical_index,
        )
        candidate_specs.append((slot, n_back, cue))
        history = train_existing_adaptive_relation_capability(
            agent,
            slot=slot,
            verifier_n_back=n_back,
            updates=candidate_updates,
            batch_size=batch_size,
            steps=steps,
            seed=seed + 6000 + index,
            learning_rate=1e-2,
            forced_slot=True,
            cue_symbol=cue,
        )
        own_score, own_bits, own_outcomes = _rollout_score(
            agent,
            n_back=n_back,
            cue_symbol=cue,
            slot=slot,
            batch_size=batch_size,
            steps=steps,
            seed=seed + 7000 + index,
            record_retention=True,
        )
        candidate_features.append(_candidate_feature(agent, slot))
        candidate_history.append(
            {
                "slot": slot,
                "logical_index": logical_index,
                "n_back": n_back,
                "own_score": own_score,
                "optimizer_updates": len(history),
            }
        )
        bits += own_bits + sum(row.unique_verifier_bits for row in history)
        outcomes += own_outcomes + len(history) * batch_size * (
            steps + 1
        )
    incoming_scores: list[float] = []
    incoming_bits = 0
    incoming_outcomes = 0
    # The incoming task is held out as logical capability zero while the two
    # physical slots are permuted. The policy must therefore learn which
    # opaque capability is disposable rather than memorizing slot order.
    incoming_n_back = CAPABILITY_N_BACKS[incoming_logical_index]
    incoming_cue = CAPABILITY_CUES[incoming_logical_index]
    for slot in (1, 2):
        scores: list[float] = []
        for index in range(probe_batches):
            score, run_bits, run_outcomes = _rollout_score(
                agent,
                n_back=incoming_n_back,
                cue_symbol=incoming_cue,
                slot=slot,
                batch_size=batch_size,
                steps=steps,
                seed=seed + 8000 + slot * 100 + index,
                record_retention=False,
            )
            scores.append(score)
            incoming_bits += run_bits
            incoming_outcomes += run_outcomes
        incoming_scores.append(sum(scores) / len(scores))
    bits += incoming_bits
    outcomes += incoming_outcomes
    with torch.no_grad():
        context = _incoming_context(agent, incoming_cue).unsqueeze(0)
        features = torch.stack(candidate_features).unsqueeze(0)
    return {
        "agent": agent,
        "context": context,
        "candidate_features": features,
        "incoming_scores": torch.tensor(incoming_scores, dtype=torch.float32),
        "candidate_history": candidate_history,
        "candidate_specs": candidate_specs,
        "incoming_n_back": incoming_n_back,
        "incoming_cue": incoming_cue,
        "base_audit": base_audit,
        "training_optimizer_updates": old_updates + 2 * candidate_updates,
        "training_logical_lifetimes": batch_size
        * (
            old_updates
            + 2 * candidate_updates
            + base_audits
            + 2
            + 2 * probe_batches
        ),
        "training_unique_verifier_bits": bits,
        "training_verifier_outcome_events": outcomes,
        "replayed_examples": sum(row.replayed_examples for row in old_history),
    }


def _policy_scores(
    policy: ExternalCapabilityEvictionPolicy,
    world: dict[str, Any],
) -> torch.Tensor:
    return policy.score_candidates(
        world["context"], world["candidate_features"]
    ).squeeze(0)


def _train_policy_world(
    policy: ExternalCapabilityEvictionPolicy,
    optimizer: torch.optim.Optimizer,
    world: dict[str, Any],
    *,
    reward_shuffle: bool,
) -> dict[str, float]:
    scores = _policy_scores(policy, world)
    utilities = 1.0 - world["incoming_scores"].unsqueeze(0)
    if reward_shuffle:
        utilities = torch.rand_like(utilities)
    loss, advantage = paired_counterfactual_ranking_loss(
        scores[1:].unsqueeze(0),
        torch.tensor([[0, 1]], dtype=torch.long),
        utilities,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [parameter for parameter in policy.parameters() if parameter.grad is not None]
    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "advantage": float(advantage.mean()),
        "incoming_disposability_gap": float(
            (utilities[:, 0] - utilities[:, 1]).mean()
        ),
    }


def _refresh_incoming_utility(
    world: dict[str, Any],
    *,
    batch_size: int,
    steps: int,
    seed: int,
    probe_batches: int,
) -> None:
    """Collect fresh verifier outcomes from an already acquired frozen world."""

    agent = world["agent"]
    incoming_scores: list[float] = []
    bits = 0
    outcomes = 0
    for slot in (1, 2):
        scores: list[float] = []
        for index in range(probe_batches):
            score, run_bits, run_outcomes = _rollout_score(
                agent,
                n_back=world["incoming_n_back"],
                cue_symbol=world["incoming_cue"],
                slot=slot,
                batch_size=batch_size,
                steps=steps,
                seed=seed + slot * 100 + index,
                record_retention=False,
            )
            scores.append(score)
            bits += run_bits
            outcomes += run_outcomes
        incoming_scores.append(sum(scores) / len(scores))
    world["incoming_scores"] = torch.tensor(incoming_scores, dtype=torch.float32)
    world["training_unique_verifier_bits"] += bits
    world["training_verifier_outcome_events"] += outcomes
    world["training_logical_lifetimes"] += batch_size * 2 * probe_batches


def _select_slot(
    policy: ExternalCapabilityEvictionPolicy,
    world: dict[str, Any],
) -> tuple[int, torch.Tensor, torch.Tensor]:
    scores = _policy_scores(policy, world)
    agent = world["agent"]
    keys = torch.stack(
        [agent.capability_address_for(slot) for slot in range(3)]
    )
    masked, protected = agent.retention.mask_eviction_scores(keys, scores)
    return int(masked.argmax()), scores, protected


def _evaluate_selection(
    policy: ExternalCapabilityEvictionPolicy,
    *,
    worlds: list[dict[str, Any]],
    corrupt_features: bool = False,
) -> dict[str, Any]:
    selected: list[int] = []
    oracle: list[int] = []
    protected_masks: list[list[bool]] = []
    for world in worlds:
        if corrupt_features:
            original = world["candidate_features"]
            world = {**world, "candidate_features": torch.zeros_like(original)}
        chosen, _scores, protected = _select_slot(policy, world)
        selected.append(chosen)
        oracle.append(1 + int(torch.argmax(1.0 - world["incoming_scores"])))
        protected_masks.append(protected.tolist())
    accuracy = sum(a == b for a, b in zip(selected, oracle, strict=True)) / len(
        worlds
    )
    return {
        "selected_slots": selected,
        "oracle_slots": oracle,
        "accuracy": accuracy,
        "protected_masks": protected_masks,
    }


def _replacement_audit(
    policy: ExternalCapabilityEvictionPolicy,
    world: dict[str, Any],
    *,
    batch_size: int,
    steps: int,
    replacement_updates: int,
    replacement_learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    agent = world["agent"]
    chosen, scores, protected = _select_slot(policy, world)
    oracle = 1 + int(torch.argmax(1.0 - world["incoming_scores"]))
    before_mastered = _bank_digest(agent, ())
    # Seed stale global route evidence for the candidate. Replacement must
    # clear this physical-slot state before the new capability is trained.
    agent.route_evidence.observe(chosen, 0.0)
    agent.route_evidence.observe(chosen, 0.0)
    before_route = agent.route_evidence.status()
    receipt = agent.replace_unprotected_adaptive_relation_capability(
        chosen,
        memory_capacity=8,
        seed=seed + 15000,
    )
    after_replacement_route = agent.route_evidence.status()
    history = train_existing_adaptive_relation_capability(
        agent,
        slot=chosen,
        verifier_n_back=world["incoming_n_back"],
        updates=replacement_updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed + 16000,
        learning_rate=replacement_learning_rate,
        forced_slot=True,
        cue_symbol=world["incoming_cue"],
    )
    replacement_score, replacement_bits, replacement_outcomes = _rollout_score(
        agent,
        n_back=world["incoming_n_back"],
        cue_symbol=world["incoming_cue"],
        slot=chosen,
        batch_size=batch_size,
        steps=steps,
        seed=seed + 17000,
        record_retention=False,
    )
    retained_slot = 3 - chosen if chosen in (1, 2) else 1
    retained_index = retained_slot - 1
    retained_score, retained_bits, retained_outcomes = _rollout_score(
        agent,
        n_back=world["candidate_specs"][retained_index][1],
        cue_symbol=world["candidate_specs"][retained_index][2],
        slot=retained_slot,
        batch_size=batch_size,
        steps=steps,
        seed=seed + 18000,
        record_retention=False,
    )
    base_score, base_bits, base_outcomes = _rollout_score(
        agent,
        n_back=BASE_N_BACK,
        cue_symbol=BASE_CUE,
        slot=0,
        batch_size=batch_size,
        steps=steps,
        seed=seed + 19000,
        record_retention=False,
    )
    route_verifier = NBackVerifier(
        batch_size=batch_size,
        n_back=world["incoming_n_back"],
        steps=steps,
        symbol_count=4,
        cue_symbol=world["incoming_cue"],
        seed=seed + 20000,
    )
    route_rollout = agent.rollout(
        route_verifier,
        sample=False,
        record_retention=False,
        context_route=True,
        record_context_route=True,
    )
    route_status = _context_status(agent, world["incoming_cue"])
    return {
        "selected_slot": chosen,
        "oracle_slot": oracle,
        "selection_correct": chosen == oracle,
        "policy_scores": scores.tolist(),
        "protected": protected.tolist(),
        "receipt": receipt,
        "stale_route_before": {
            "attempts": list(before_route.attempts),
            "successes": list(before_route.successes),
            "preferred_slot": before_route.preferred_slot,
        },
        "stale_route_after": {
            "attempts": list(after_replacement_route.attempts),
            "successes": list(after_replacement_route.successes),
            "preferred_slot": after_replacement_route.preferred_slot,
        },
        "stale_route_cleared": after_replacement_route.attempts[chosen] == 0
        and after_replacement_route.successes[chosen] == 0.0
        and after_replacement_route.preferred_slot != chosen,
        "replacement_fresh": replacement_score,
        "retained_capability": retained_score,
        "base_retention": base_score,
        "automatic_route_score": float(route_rollout.eligible_accuracy.mean()),
        "route_status": route_status,
        "prior_mastered_bank_unchanged": before_mastered == _bank_digest(agent, ()),
        "optimizer_updates": len(history),
        "unique_verifier_bits": replacement_bits
        + retained_bits
        + base_bits
        + batch_size * route_verifier.eligible_trials,
        "verifier_outcome_events": replacement_outcomes
        + retained_outcomes
        + base_outcomes
        + batch_size * route_verifier.steps,
        "unique_logical_lifetimes": batch_size * 4,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.train_worlds,
        args.eval_worlds,
        args.old_updates,
        args.candidate_updates,
        args.replacement_updates,
        args.probe_batches,
        args.policy_steps_per_world,
        args.base_audits,
        args.batch_size,
        args.steps,
    ) < 1:
        raise ValueError("all budgets must be positive")
    policy = ExternalCapabilityEvictionPolicy(
        context_width=32,
        candidate_width=CANDIDATE_FEATURE_WIDTH,
        hidden=args.policy_hidden,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_learning_rate)
    history: list[dict[str, float | int]] = []
    training_bits = 0
    training_outcomes = 0
    training_lifetimes = 0
    training_updates = 0
    replayed_examples = 0
    training_worlds: list[dict[str, Any]] = []
    for world_index in range(args.train_worlds):
        world = _build_world(
            seed=args.seed + world_index * 101,
            batch_size=args.batch_size,
            steps=args.steps,
            old_updates=args.old_updates,
            candidate_updates=args.candidate_updates,
            strong_first=(world_index // 2) % 2 == 0,
            incoming_logical_index=world_index % 2,
            probe_batches=args.probe_batches,
            base_audits=args.base_audits,
        )
        training_worlds.append(world)
        training_bits += int(world["training_unique_verifier_bits"])
        training_outcomes += int(world["training_verifier_outcome_events"])
        training_lifetimes += int(world["training_logical_lifetimes"])
        training_updates += int(world["training_optimizer_updates"])
        replayed_examples += int(world["replayed_examples"])
        for policy_step in range(args.policy_steps_per_world):
            _refresh_incoming_utility(
                world,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=args.seed
                + world_index * 100000
                + policy_step * 1000,
                probe_batches=1,
            )
            result = _train_policy_world(
                policy, optimizer, world, reward_shuffle=False
            )
            history.append(
                {
                    "world": world_index + 1,
                    "policy_step": policy_step + 1,
                    **result,
                }
            )
            training_bits += int(
                args.batch_size
                * 2
                * (args.steps - 1 - world["incoming_n_back"])
            )
            training_outcomes += int(args.batch_size * 2 * (args.steps + 1))
            training_lifetimes += int(args.batch_size * 2)
            training_updates += 1
    eval_worlds = [
        _build_world(
            seed=args.seed + 10000 + world_index * 101,
            batch_size=args.batch_size,
            steps=args.steps,
            old_updates=args.old_updates,
            candidate_updates=args.candidate_updates,
            strong_first=(world_index // 2) % 2 == 0,
            incoming_logical_index=world_index % 2,
            probe_batches=args.probe_batches,
            base_audits=args.base_audits,
        )
        for world_index in range(args.eval_worlds)
    ]
    selection = _evaluate_selection(policy, worlds=eval_worlds)
    corrupted_selection = _evaluate_selection(
        policy, worlds=eval_worlds, corrupt_features=True
    )
    shuffled_policy = ExternalCapabilityEvictionPolicy(
        context_width=32,
        candidate_width=CANDIDATE_FEATURE_WIDTH,
        hidden=args.policy_hidden,
    )
    shuffled_optimizer = torch.optim.Adam(
        shuffled_policy.parameters(), lr=args.policy_learning_rate
    )
    shuffled_history: list[dict[str, float | int]] = []
    for world_index, world in enumerate(training_worlds):
        for policy_step in range(args.policy_steps_per_world):
            _refresh_incoming_utility(
                world,
                batch_size=args.batch_size,
                steps=args.steps,
                seed=args.seed
                + 300000
                + world_index * 100000
                + policy_step * 1000,
                probe_batches=1,
            )
            result = _train_policy_world(
                shuffled_policy,
                shuffled_optimizer,
                world,
                reward_shuffle=True,
            )
            shuffled_history.append(
                {
                    "world": world_index + 1,
                    "policy_step": policy_step + 1,
                    **result,
                }
            )
            training_bits += int(
                args.batch_size
                * 2
                * (args.steps - 1 - world["incoming_n_back"])
            )
            training_outcomes += int(args.batch_size * 2 * (args.steps + 1))
            training_lifetimes += int(args.batch_size * 2)
    shuffled_selection = _evaluate_selection(shuffled_policy, worlds=eval_worlds)
    replacement = _replacement_audit(
        policy,
        eval_worlds[0],
        batch_size=args.batch_size,
        steps=args.steps,
        replacement_updates=args.replacement_updates,
        replacement_learning_rate=args.replacement_learning_rate,
        seed=args.seed,
    )
    evaluation_bits = sum(
        int(world["training_unique_verifier_bits"]) for world in eval_worlds
    )
    evaluation_outcomes = sum(
        int(world["training_verifier_outcome_events"]) for world in eval_worlds
    )
    evaluation_lifetimes = sum(
        int(world["training_logical_lifetimes"]) for world in eval_worlds
    )
    all_protected = bool(eval_worlds[0]["agent"].retention.status(
        eval_worlds[0]["agent"].capability_address_for(0)
    ).protected)
    controller_frozen = all(
        not parameter.requires_grad
        for parameter in eval_worlds[0]["agent"].controller.parameters()
    )
    promotion_gates = {
        "learned_selection": selection["accuracy"] >= 0.75,
        "reward_shuffle_chance": shuffled_selection["accuracy"] <= 0.67,
        "feature_corruption_chance": corrupted_selection["accuracy"] <= 0.67,
        "protected_mastery_mask": all_protected
        and replacement["selected_slot"] in (1, 2),
        "replacement_fresh": replacement["replacement_fresh"] >= 0.8,
        "retained_capability": replacement["retained_capability"] >= 0.8,
        "base_retention": replacement["base_retention"] >= 0.8,
        "prior_mastered_bank_unchanged": replacement[
            "prior_mastered_bank_unchanged"
        ],
        "stale_route_cleared": replacement["stale_route_cleared"],
        "zero_replay": replayed_examples == 0,
        "controller_frozen": controller_frozen,
    }
    report = {
        "schema": "neural-computer.brainworkshop-learned-eviction.v1",
        "status": (
            "promoted_learned_capability_eviction"
            if all(promotion_gates.values())
            else "unpromoted_learned_capability_eviction"
        ),
        "fixed_core_seed": FIXED_CORE_SEED,
        "train_worlds": args.train_worlds,
        "eval_worlds": args.eval_worlds,
        "policy": policy.configuration(),
        "learner_visible_inputs": [
            "opaque incoming learned event tensor",
            "opaque capability address tensor",
            "deterministic scalar verifier outcomes",
        ],
        "history": history,
        "evaluation_worlds": [
            {
                "candidate_history": world["candidate_history"],
                "incoming_scores": world["incoming_scores"].tolist(),
                "base_audit": world["base_audit"],
            }
            for world in eval_worlds
        ],
        "selection": selection,
        "corrupted_selection": corrupted_selection,
        "reward_shuffle_control": shuffled_selection,
        "reward_shuffle_history": shuffled_history,
        "replacement": replacement,
        "promotion_gates": promotion_gates,
        "unique_verifier_bits": training_bits
        + evaluation_bits
        + int(replacement["unique_verifier_bits"]),
        "unique_logical_lifetimes": training_lifetimes
        + evaluation_lifetimes
        + int(replacement["unique_logical_lifetimes"]),
        "optimizer_updates": training_updates + int(replacement["optimizer_updates"]),
        "replayed_examples": replayed_examples,
        "verifier_outcome_events": training_outcomes
        + evaluation_outcomes
        + int(replacement["verifier_outcome_events"]),
        "feedback_events": training_bits
        + evaluation_bits
        + int(replacement["unique_verifier_bits"]),
        "claim_boundary": (
            "Promoted learned outcome-only utility selection for a bounded "
            "canonical capability bank only; retention masking remains explicit. "
            "This does not establish persistent learned consolidation, unbounded "
            "memory, arbitrary new computation, or general continual learning."
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_experiment(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
