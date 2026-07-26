"""Adapt memory utility from reward while the same physical banks persist."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_multifeature_utility import _materialize_histories
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_physical_stream import (
    _apply_winner,
    _candidate_batch,
    _initial_rows,
    _physical_rewards,
    _stream_features,
    _tensor_rewards,
)
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_multifeature_utility_adaptation import _expanded_controller
from .strategy_memory import (
    LatentStrategyMemory,
    VerifierTrainedContextEncoder,
    physical_context_key,
)


def _curriculum_phases(
        curriculum: str,
        rounds_per_phase: int,
        ) -> list[tuple[str, tuple[float, float, float], int]]:
    if curriculum == "context_reliability_ramp":
        # Six distinct contexts at the same six-round cost as the usual
        # three-phase, two-round screen. Only reliability changes.
        reliability = (0.0, 0.1, 0.2, 0.3, 0.4, 0.0)
        return [
            (
                f"context_reliability_{weight:.1f}",
                ((1.0 - weight) / 2, (1.0 - weight) / 2, weight),
                1,
            )
            for weight in reliability
        ]
    if curriculum == "gradual_reliability":
        phases = [
            ("mild_reliability", (0.35, 0.35, 0.3)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("mild_reliability_return", (0.35, 0.35, 0.3)),
        ]
    elif curriculum == "interleaved_reliability":
        # A short prefix must exercise all three contexts.  This changes only
        # their temporal ordering relative to ``standard``: the same weights
        # and the same total number of physical rounds are retained.
        cycle = [
            ("old_equal", (0.5, 0.5, 0.0)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("old_return", (0.5, 0.5, 0.0)),
        ]
        return [
            (phase, weights, 1)
            for _ in range(rounds_per_phase)
            for phase, weights in cycle
        ]
    elif curriculum == "cyclic_reliability_blocks6":
        if rounds_per_phase % 6:
            raise ValueError(
                "cyclic_reliability_blocks6 requires rounds-per-phase "
                "to be divisible by six")
        # Intermediate temporal-contiguity control: retain six consecutive
        # experiences from one context, then revisit all contexts. At a fixed
        # total budget this exposes return sooner than the blocked schedule.
        cycle = [
            ("old_equal", (0.5, 0.5, 0.0)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("old_return", (0.5, 0.5, 0.0)),
        ]
        return [
            (phase, weights, 6)
            for _ in range(rounds_per_phase // 6)
            for phase, weights in cycle
        ]
    else:
        phases = [
            ("old_equal", (0.5, 0.5, 0.0)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("old_return", (0.5, 0.5, 0.0)),
        ]
    return [
        (phase, weights, rounds_per_phase)
        for phase, weights in phases
    ]


def _value_diverse_admission(
        memory: LatentStrategyMemory,
        candidates: list[torch.Tensor],
        rewards: list[float],
        ) -> tuple[int, int | None]:
    """Choose the scored candidate/slot maximizing latent bank separation."""
    flattened = [candidate.flatten() for candidate in candidates]
    if memory.count == 0:
        return max(range(len(rewards)), key=rewards.__getitem__), None
    if memory.count < memory.capacity:
        distances = [
            float(torch.cdist(
                candidate.unsqueeze(0),
                memory.values[:memory.count]).min())
            for candidate in flattened
        ]
        index = max(
            range(len(flattened)),
            key=lambda candidate: (distances[candidate], rewards[candidate]))
        return index, None
    choices = []
    for candidate_index, candidate in enumerate(flattened):
        for slot in range(memory.count):
            proposed = memory.values[:memory.count].clone()
            proposed[slot] = candidate
            separation = float(torch.pdist(proposed).min())
            choices.append((
                separation, rewards[candidate_index],
                candidate_index, slot))
    _, _, candidate_index, slot = max(choices)
    return candidate_index, slot


def _clone_model(
        payload: dict[str, object], device: torch.device,
        ) -> UnifiedCognitiveController:
    model, _ = _expanded_controller(payload, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7031)
    parser.add_argument(
        "--experience-seed", type=int,
        help="override only physical histories and candidate batches")
    parser.add_argument(
        "--policy-perturbation-seed", type=int,
        help="override only the two-parameter policy horse race")
    parser.add_argument(
        "--context-proposal-seed", type=int,
        help="override only context-metric proposal directions")
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--rounds-per-phase", type=int, default=3)
    parser.add_argument(
        "--max-physical-rounds", type=int,
        help=(
            "stop an otherwise unchanged curriculum at an exact prefix; "
            "prefix reports can never pass the graduation gate"))
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--shuffle-physical-rewards", action="store_true")
    parser.add_argument(
        "--reset-banks-each-round", action="store_true",
        help="fresh-bank control: rematerialize physical histories each round")
    parser.add_argument(
        "--target-intervention",
        choices=(
            "none", "cold", "empty_history", "shuffled_history",
            "shuffled_strategy_keys"),
        default="none",
        help="one-time intervention immediately before reliability transfer")
    parser.add_argument(
        "--curriculum",
        choices=(
            "standard", "gradual_reliability",
            "context_reliability_ramp", "interleaved_reliability",
            "cyclic_reliability_blocks6"),
        default="standard")
    parser.add_argument(
        "--strategy-memory-capacity", type=int, default=0,
        help="zero uses the global residual; positive enables latent RAM")
    parser.add_argument(
        "--strategy-admission",
        choices=(
            "winner", "action_diversity", "value_diversity"),
        default="winner",
        help=(
            "action_diversity preserves a behaviorally novel already-scored "
            "candidate without adding verifier evaluations"))
    parser.add_argument(
        "--context-learning-rate", type=float, default=0.0,
        help="positive enables verifier-trained context feature weighting")
    parser.add_argument(
        "--soft-context-perturbation", type=float, default=0.0,
        help="SPSA radius; positive evaluates two soft retrieval mixtures")
    parser.add_argument("--soft-context-temperature", type=float, default=0.25)
    parser.add_argument(
        "--soft-context-direction-proposals", type=int, default=1,
        help=(
            "cost-free latent directions screened by action disagreement "
            "before evaluating one pair with the physical verifier"))
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.banks < 2 or args.rounds_per_phase < 1
            or args.soft_context_direction_proposals < 1
            or (
                args.max_physical_rounds is not None
                and args.max_physical_rounds < 1)):
        raise ValueError("at least two banks and one round are required")

    seed_everything(args.seed)
    experience_seed = (
        args.seed if args.experience_seed is None
        else args.experience_seed)
    policy_perturbation_seed = (
        args.seed if args.policy_perturbation_seed is None
        else args.policy_perturbation_seed)
    context_proposal_seed = (
        args.seed if args.context_proposal_seed is None
        else args.context_proposal_seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = _expanded_controller(payload, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = _clone_model(payload, device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    initial_residual = (
        model.memory_replacement_extra_gate.weight.detach().clone())
    phases = _curriculum_phases(
        args.curriculum, args.rounds_per_phase)
    started = time.perf_counter()
    initial = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=experience_seed * 10_000, device=device,
        write_threshold=0.5, noise_scale=0.04,
        recency_weight=0.5, frequency_weight=0.5)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        initial_directory = root / "initial"
        initial_directory.mkdir()
        memories, _, _, _, initial_exact, requested_exact = (
            _materialize_histories(
                model, initial, initial_directory, device=device))
        row_batch, row_queries = _initial_rows(
            initial, banks=args.banks, capacity=args.bank_capacity,
            device=device)
        direction_generator = torch.Generator(device=device).manual_seed(
            policy_perturbation_seed + 70_000_000)
        context_direction_generator = torch.Generator(
            device=device).manual_seed(
                context_proposal_seed + 71_000_000)
        strategy_memory = (
            LatentStrategyMemory(capacity=args.strategy_memory_capacity,
                                 key_width=13, device=device)
            if args.strategy_memory_capacity > 0 else None)
        context_encoder = (
            VerifierTrainedContextEncoder(width=13).to(device)
            if strategy_memory is not None else None)
        context_optimizer = (
            torch.optim.Adam(
                context_encoder.parameters(),
                lr=args.context_learning_rate)
            if context_encoder is not None
            and args.context_learning_rate > 0 else None)
        context_updates = 0
        context_losses = []
        context_advantages = []
        previous_reward_signature = torch.zeros(3, device=device)
        strategy_save_reloads = 0
        strategy_candidate_evaluations = 0
        trace = []
        phase_rows_by_name: dict[str, list[dict[str, object]]] = {}
        phase_weights_by_name: dict[str, tuple[float, float, float]] = {}
        state_exact = initial_exact
        candidate_exact = 0
        transition_exact = True
        maximum_parity_difference = 0.0
        maximum_cross_choice_regret = 0.0
        total_replacements = 0
        rounds = 0
        soft_context_pair_count = 0
        soft_context_action_divergent_pairs = 0
        soft_context_reward_divergent_pairs = 0
        soft_context_reward_delta_sum = 0.0
        soft_context_direction_proposals_screened = 0
        soft_context_selected_preverifier_action_disagreements = 0
        target_intervention_applied = False

        for phase_index, (phase, weights, phase_rounds) in enumerate(phases):
            if (
                    args.max_physical_rounds is not None
                    and rounds >= args.max_physical_rounds):
                break
            is_peak_reliability = (
                weights[2] == max(item[1][2] for item in phases))
            intervention_active = (
                is_peak_reliability and not target_intervention_applied)
            if intervention_active:
                if args.target_intervention == "cold":
                    model.memory_replacement_extra_gate.weight.copy_(
                        initial_residual)
                    if strategy_memory is not None:
                        strategy_memory = LatentStrategyMemory(
                            capacity=args.strategy_memory_capacity,
                            key_width=13, device=device)
                        previous_reward_signature.zero_()
                elif (
                        args.target_intervention
                        == "shuffled_strategy_keys"
                        and strategy_memory is not None
                        and strategy_memory.count > 1):
                        strategy_memory.keys[:strategy_memory.count] = (
                            strategy_memory.keys[:strategy_memory.count].roll(
                            1, dims=0))
                target_intervention_applied = True
            phase_rows = []
            for round_index in range(phase_rounds):
                if (
                        args.max_physical_rounds is not None
                        and rounds >= args.max_physical_rounds):
                    break
                rounds += 1
                seed = (
                    experience_seed * 10_000_000
                    + phase_index * 1_000_000 + round_index + 1)
                data = frequency_recency_batch(
                    model, banks=args.banks,
                    capacity=args.bank_capacity, seed=seed,
                    device=device, write_threshold=0.5,
                    noise_scale=0.04,
                    recency_weight=weights[0],
                    frequency_weight=weights[1],
                    reliability_weight=weights[2])
                if args.reset_banks_each_round:
                    fresh_directory = (
                        root / f"round-{rounds:03d}-fresh-initial")
                    fresh_directory.mkdir()
                    (
                        memories, _, _, _, exact, fresh_requested_exact,
                    ) = _materialize_histories(
                        model, data, fresh_directory, device=device)
                    state_exact += exact
                    requested_exact += fresh_requested_exact
                    row_batch, row_queries = _initial_rows(
                        data, banks=args.banks,
                        capacity=args.bank_capacity, device=device)
                candidate_batch = _candidate_batch(
                    data, banks=args.banks,
                    capacity=args.bank_capacity, device=device)
                candidate_queries = data["query_group"][:, -1]
                generator = torch.Generator(
                    device=device).manual_seed(seed + 70_000_000)
                noise = (
                    torch.rand(
                        args.banks, args.bank_capacity,
                        generator=generator, device=device) * 2 - 1
                ) * 0.04
                features, target = _stream_features(
                    model, memories, data["candidate_key"],
                    data["candidate_strength"], weights=weights,
                    noise=noise)
                if (
                        intervention_active
                        and args.target_intervention in (
                            "empty_history", "shuffled_history")):
                    visible_memories = []
                    for source in memories:
                        visible = DiskLatentMemory.__new__(
                            DiskLatentMemory)
                        visible.store = source.store.clone()
                        for field in (
                                "access_count", "success_count",
                                "failure_count"):
                            values = getattr(visible.store, field)
                            if args.target_intervention == "empty_history":
                                values[:visible.count].zero_()
                            else:
                                values[:visible.count] = values[
                                    :visible.count].roll(1)
                        visible_memories.append(visible)
                    features, _ = _stream_features(
                        model, visible_memories, data["candidate_key"],
                        data["candidate_strength"], weights=weights,
                        noise=noise)
                context_key = physical_context_key(
                    features, previous_reward_signature)
                strategy_retrieval = None
                strategy_slot_unique_action_patterns = 0
                strategy_slot_max_action_disagreements = 0
                if strategy_memory is None:
                    current = (
                        model.memory_replacement_extra_gate.weight
                        .detach().clone())
                else:
                    slot_actions = []
                    for slot in range(strategy_memory.count):
                        model.memory_replacement_extra_gate.weight.copy_(
                            strategy_memory.values[slot].reshape(1, -1))
                        slot_actions.append(
                            model.memory_replacement_scores(
                                features).argmax(-1))
                    if slot_actions:
                        strategy_slot_unique_action_patterns = len({
                            tuple(actions.tolist())
                            for actions in slot_actions
                        })
                        strategy_slot_max_action_disagreements = max(
                            (
                                int((left != right).sum())
                                for left in slot_actions
                                for right in slot_actions
                            ),
                            default=0)
                    strategy_retrieval = strategy_memory.retrieve(
                        context_key, initial_residual.flatten(),
                        encoder=context_encoder)
                    current = strategy_retrieval.value.reshape(1, -1)
                    model.memory_replacement_extra_gate.weight.copy_(current)
                direction = torch.randint(
                    0, 2, (2,), generator=direction_generator,
                    device=device).float() * 2 - 1
                signs = (1.0, 0.0, -1.0)
                physical_means = []
                tensor_means = []
                actions_by_candidate = []
                candidate_weights = []
                for candidate_index, sign in enumerate(signs):
                    candidate_weight = (
                        current + sign * args.perturbation
                        * direction.unsqueeze(0))
                    candidate_weights.append(candidate_weight)
                    model.memory_replacement_extra_gate.weight.copy_(
                        candidate_weight)
                    actions = model.memory_replacement_scores(
                        features).argmax(-1)
                    actions_by_candidate.append(actions)
                    directory = (
                        root / f"round-{rounds:03d}-candidate-"
                        f"{candidate_index}")
                    directory.mkdir()
                    physical, exact = _physical_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        directory, device=device)
                    tensor = _tensor_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        device=device)
                    candidate_exact += exact
                    physical_means.append(float(physical.mean()))
                    tensor_means.append(float(tensor.mean()))
                post_probe_retrieval = None
                post_probe_weight = None
                soft_retrievals = []
                soft_weights = []
                context_direction = None
                if strategy_memory is not None and strategy_memory.count:
                    reward_signature = torch.tensor(
                        physical_means, device=device)
                    reward_context_key = physical_context_key(
                        features, reward_signature)
                    if args.soft_context_perturbation > 0:
                        original_scale = (
                            context_encoder.log_scale.detach().clone())
                        proposals = []
                        for _ in range(
                                args.soft_context_direction_proposals):
                            proposed_direction = torch.randint(
                                0, 2, (13,),
                                generator=context_direction_generator,
                                device=device).float() * 2 - 1
                            proposed_retrievals = []
                            proposed_actions = []
                            for context_sign in (1.0, -1.0):
                                context_encoder.log_scale.data.copy_(
                                    original_scale
                                    + context_sign
                                    * args.soft_context_perturbation
                                    * proposed_direction)
                                retrieval = strategy_memory.retrieve_soft(
                                    reward_context_key, current.flatten(),
                                    encoder=context_encoder,
                                    temperature=args.soft_context_temperature)
                                proposed_retrievals.append(retrieval)
                                model.memory_replacement_extra_gate.weight.copy_(
                                    retrieval.value.reshape(1, -1))
                                proposed_actions.append(
                                    model.memory_replacement_scores(
                                        features).argmax(-1))
                            disagreement = int(
                                (
                                    proposed_actions[0]
                                    != proposed_actions[1]
                                ).sum())
                            proposals.append((
                                disagreement, proposed_direction,
                                proposed_retrievals))
                        soft_context_direction_proposals_screened += len(
                            proposals)
                        (
                            selected_disagreement,
                            context_direction,
                            soft_retrievals,
                        ) = max(proposals, key=lambda item: item[0])
                        soft_context_selected_preverifier_action_disagreements += (
                            selected_disagreement)
                        context_encoder.log_scale.data.copy_(original_scale)
                    else:
                        soft_retrievals.append(strategy_memory.retrieve(
                            reward_context_key, current.flatten(),
                            encoder=context_encoder))
                    for retrieval_index, retrieval in enumerate(
                            soft_retrievals):
                        strategy_candidate_evaluations += 1
                        candidate_weight = retrieval.value.reshape(1, -1)
                        soft_weights.append(candidate_weight)
                        candidate_weights.append(candidate_weight)
                        model.memory_replacement_extra_gate.weight.copy_(
                            candidate_weight)
                        memory_actions = model.memory_replacement_scores(
                            features).argmax(-1)
                        actions_by_candidate.append(memory_actions)
                        memory_directory = root / (
                            f"round-{rounds:03d}-candidate-memory-"
                            f"{retrieval_index}")
                        memory_directory.mkdir()
                        physical, exact = _physical_rewards(
                            model, memories, data, row_batch, row_queries,
                            candidate_batch, candidate_queries,
                            memory_actions, target, memory_directory,
                            device=device)
                        tensor = _tensor_rewards(
                            model, memories, data, row_batch, row_queries,
                            candidate_batch, candidate_queries,
                            memory_actions, target, device=device)
                        candidate_exact += exact
                        physical_means.append(float(physical.mean()))
                        tensor_means.append(float(tensor.mean()))
                    post_probe_retrieval = soft_retrievals[0]
                    post_probe_weight = soft_weights[0]
                    if len(soft_retrievals) == 2:
                        soft_context_pair_count += 1
                        positive_actions = actions_by_candidate[-2]
                        negative_actions = actions_by_candidate[-1]
                        soft_context_action_divergent_pairs += int(
                            not torch.equal(
                                positive_actions, negative_actions))
                        reward_delta = abs(
                            physical_means[-2] - physical_means[-1])
                        soft_context_reward_delta_sum += reward_delta
                        soft_context_reward_divergent_pairs += int(
                            reward_delta > 1e-9)
                aligned_rewards = list(physical_means)
                if args.shuffle_physical_rewards:
                    order = torch.randperm(
                        len(physical_means),
                        generator=direction_generator,
                        device=device).tolist()
                    aligned_rewards = [
                        physical_means[index] for index in order]
                physical_winner = max(
                    range(len(aligned_rewards)),
                    key=aligned_rewards.__getitem__)
                tensor_winner = max(
                    range(len(tensor_means)),
                    key=tensor_means.__getitem__)
                maximum_parity_difference = max(
                    maximum_parity_difference,
                    max(abs(a - b) for a, b in zip(
                        physical_means, tensor_means)))
                cross_regret = max(
                    max(tensor_means) - tensor_means[physical_winner],
                    max(physical_means) - physical_means[tensor_winner])
                maximum_cross_choice_regret = max(
                    maximum_cross_choice_regret, cross_regret)
                if (
                        context_direction is not None
                        and len(soft_retrievals) == 2
                        and args.context_learning_rate > 0):
                    context_advantages.append(context_encoder.spsa_step(
                        context_direction,
                        positive_reward=physical_means[3],
                        negative_reward=physical_means[4],
                        step_size=args.context_learning_rate))
                    context_updates += 1
                elif (
                        context_optimizer is not None
                        and post_probe_retrieval is not None
                        and post_probe_retrieval.slot is not None):
                    context_losses.append(context_encoder.reinforce(
                        reward_context_key,
                        strategy_memory.keys[:strategy_memory.count],
                        selected_slot=post_probe_retrieval.slot,
                        verified_improvement=(
                            physical_means[-1] - physical_means[1]),
                        optimizer=context_optimizer))
                    context_updates += 1
                if physical_winner < len(signs):
                    winner_sign = signs[physical_winner]
                    winner_weight = (
                        current + winner_sign * args.step_size
                        * direction.unsqueeze(0))
                else:
                    winner_weight = soft_weights[
                        physical_winner - len(signs)]
                model.memory_replacement_extra_gate.weight.copy_(
                    winner_weight)
                strategy_slot = None
                strategy_admission_candidate = physical_winner
                strategy_admission_was_novel = None
                if strategy_memory is not None:
                    preferred_slot = None
                    if args.strategy_admission == "action_diversity":
                        existing_patterns = [
                            tuple(actions.tolist())
                            for actions in slot_actions
                        ]
                        ranked_candidates = sorted(
                            range(len(physical_means)),
                            key=physical_means.__getitem__, reverse=True)
                        novel = [
                            index for index in ranked_candidates
                            if tuple(
                                actions_by_candidate[index].tolist()
                            ) not in existing_patterns
                        ]
                        if novel:
                            strategy_admission_candidate = novel[0]
                            strategy_admission_was_novel = True
                            if strategy_memory.count == strategy_memory.capacity:
                                duplicate_slots = [
                                    slot for slot, pattern in enumerate(
                                        existing_patterns)
                                    if existing_patterns.count(pattern) > 1
                                ]
                                if duplicate_slots:
                                    reliability = (
                                        (
                                            strategy_memory.success.float()
                                            + 1
                                        ) / (
                                            strategy_memory.success
                                            + strategy_memory.failure + 2
                                        ).float())
                                    utility = (
                                        strategy_memory.usage.float()
                                        + reliability)
                                    preferred_slot = min(
                                        duplicate_slots,
                                        key=lambda slot: float(
                                            utility[slot]))
                        else:
                            strategy_admission_was_novel = False
                    elif args.strategy_admission == "value_diversity":
                        (
                            strategy_admission_candidate,
                            preferred_slot,
                        ) = _value_diverse_admission(
                            strategy_memory, candidate_weights,
                            physical_means)
                        strategy_admission_was_novel = True
                    storage_weight = candidate_weights[
                        strategy_admission_candidate]
                    reward_signature = torch.tensor(
                        physical_means[:3], device=device)
                    storage_context_key = physical_context_key(
                        features, reward_signature)
                    strategy_slot = strategy_memory.upsert(
                        storage_context_key,
                        storage_weight.flatten(),
                        verified_improvement=(
                            physical_means[strategy_admission_candidate]
                            - physical_means[1]),
                        preferred_slot=preferred_slot)
                    strategy_path = (
                        root / f"round-{rounds:03d}-strategy.pt")
                    strategy_memory.save(strategy_path)
                    restored_strategy = LatentStrategyMemory.load(
                        strategy_path, device=device)
                    strategy_save_reloads += int(
                        restored_strategy.count == strategy_memory.count
                        and torch.equal(
                            restored_strategy.keys, strategy_memory.keys)
                        and torch.equal(
                            restored_strategy.values,
                            strategy_memory.values)
                        and torch.equal(
                            restored_strategy.usage,
                            strategy_memory.usage)
                        and torch.equal(
                            restored_strategy.success,
                            strategy_memory.success)
                        and torch.equal(
                            restored_strategy.failure,
                            strategy_memory.failure))
                    strategy_memory = restored_strategy
                    previous_reward_signature = reward_signature
                learned_actions = actions_by_candidate[physical_winner]
                frozen_actions = frozen.memory_replacement_scores(
                    features).argmax(-1)
                frozen_directory = root / f"round-{rounds:03d}-frozen"
                frozen_directory.mkdir()
                frozen_rewards, exact = _physical_rewards(
                    frozen, memories, data, row_batch, row_queries,
                    candidate_batch, candidate_queries, frozen_actions,
                    target, frozen_directory, device=device)
                candidate_exact += exact
                state_directory = root / f"round-{rounds:03d}-state"
                state_directory.mkdir()
                (
                    memories, row_batch, row_queries, exact, replacements,
                    exact_transition,
                ) = _apply_winner(
                    model, memories, data, row_batch, row_queries,
                    candidate_batch, candidate_queries, learned_actions,
                    target, state_directory, device=device)
                state_exact += exact
                transition_exact = transition_exact and exact_transition
                total_replacements += replacements
                row = {
                    "phase": phase,
                    "round": round_index + 1,
                    "physical_rewards": physical_means,
                    "aligned_physical_rewards": aligned_rewards,
                    "tensor_rewards": tensor_means,
                    "physical_winner": physical_winner,
                    "learned_reward": physical_means[physical_winner],
                    "frozen_reward": float(frozen_rewards.mean()),
                    "learned_target_rate": float(
                        (learned_actions == target).float().mean()),
                    "frozen_target_rate": float(
                        (frozen_actions == target).float().mean()),
                    "residual_weights": (
                        model.memory_replacement_extra_gate.weight
                        .flatten().tolist()),
                    "strategy_retrieved_slot": (
                        strategy_retrieval.slot
                        if strategy_retrieval is not None else None),
                    "strategy_retrieval_similarity": (
                        strategy_retrieval.similarity
                        if strategy_retrieval is not None else None),
                    "strategy_updated_slot": strategy_slot,
                    "strategy_admission_candidate":
                        strategy_admission_candidate,
                    "strategy_admission_was_novel":
                        strategy_admission_was_novel,
                    "post_probe_strategy_slot": (
                        post_probe_retrieval.slot
                        if post_probe_retrieval is not None else None),
                    "post_probe_strategy_similarity": (
                        post_probe_retrieval.similarity
                        if post_probe_retrieval is not None else None),
                    "post_probe_strategy_won":
                        physical_winner >= len(signs),
                    "soft_context_candidate_count": len(soft_retrievals),
                    "soft_context_direction_proposals":
                        args.soft_context_direction_proposals,
                    "soft_context_mixture_weights": [
                        (
                            retrieval.mixture_weights.tolist()
                            if retrieval.mixture_weights is not None else None)
                        for retrieval in soft_retrievals],
                    "soft_context_actions_diverged": (
                        not torch.equal(
                            actions_by_candidate[-2],
                            actions_by_candidate[-1])
                        if len(soft_retrievals) == 2 else None),
                    "soft_context_reward_delta": (
                        abs(physical_means[-2] - physical_means[-1])
                        if len(soft_retrievals) == 2 else None),
                    "context_encoder_updates": context_updates,
                    "strategy_memory_count": (
                        strategy_memory.count
                        if strategy_memory is not None else 0),
                    "strategy_slot_unique_action_patterns":
                        strategy_slot_unique_action_patterns,
                    "strategy_slot_max_action_disagreements":
                        strategy_slot_max_action_disagreements,
                    "replacements": replacements,
                }
                trace.append(row)
                phase_rows.append(row)
            if phase_rows:
                phase_rows_by_name.setdefault(phase, []).extend(phase_rows)
                phase_weights_by_name.setdefault(phase, weights)

        phase_summaries = []
        for phase, phase_rows in phase_rows_by_name.items():
            weights = phase_weights_by_name[phase]
            phase_summaries.append({
                "phase": phase,
                "weights": weights,
                "learned_reward": sum(
                    row["learned_reward"] for row in phase_rows)
                    / len(phase_rows),
                "frozen_reward": sum(
                    row["frozen_reward"] for row in phase_rows)
                    / len(phase_rows),
                "learned_target_rate": sum(
                    row["learned_target_rate"] for row in phase_rows)
                    / len(phase_rows),
                "frozen_target_rate": sum(
                    row["frozen_target_rate"] for row in phase_rows)
                    / len(phase_rows),
                "first_round_learned_target_rate":
                    phase_rows[0]["learned_target_rate"],
                "last_round_learned_target_rate":
                    phase_rows[-1]["learned_target_rate"],
            })

        binary = evaluate(
            model, count=512, trials=6, seed=args.seed + 91_000_000,
            device=device, task="binary_mapping", feedback_trials=1)
        four_rule = evaluate(
            model, count=512, trials=6, seed=args.seed + 92_000_000,
            device=device, task="four_rule", feedback_trials=2)
        changed = [
            name for name, value in model.state_dict().items()
            if not torch.equal(
                initial_state[name], value.detach().cpu())]
        expected_state = args.banks * (
            rounds + 1
            + (rounds if args.reset_banks_each_round else 0))
        expected_candidates = args.banks * (
            4 * rounds + strategy_candidate_evaluations)
        reliability = max(
            phase_summaries, key=lambda summary: summary["weights"][2])
        old_return = phase_summaries[-1]
        planned_rounds = sum(item[2] for item in phases)
        complete_curriculum = rounds == planned_rounds
        gate = {
            "complete_curriculum": complete_curriculum,
            "every_state_save_reload_exact":
                state_exact == expected_state,
            "every_candidate_remained_bounded":
                candidate_exact == expected_candidates,
            "every_history_transition_exact": transition_exact,
            "physical_tensor_rewards_match_within_1e_6":
                maximum_parity_difference <= 1e-6,
            "physical_tensor_choices_equivalent_within_1e_6":
                maximum_cross_choice_regret <= 1e-6,
            "reliability_switch_improves_or_matches_frozen_reward":
                reliability["learned_reward"]
                >= reliability["frozen_reward"] - 1e-6,
            "reliability_switch_improves_or_matches_frozen_target_rate":
                reliability["learned_target_rate"]
                >= reliability["frozen_target_rate"],
            "old_utility_recovery_not_worse_than_frozen_by_over_13_points":
                old_return["learned_target_rate"]
                >= old_return["frozen_target_rate"] - 0.13,
            "at_least_one_replacement_persisted":
                total_replacements > 0,
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_extra_residual_changed":
                set(changed).issubset({
                    "memory_replacement_extra_gate.weight"}),
            "strategy_memory_save_reload_exact": (
                strategy_memory is None
                or strategy_save_reloads == rounds),
        }
        gate["accepted"] = (
            all(gate.values()) and not args.shuffle_physical_rewards)

    report = {
        "schema":
            "unified-controller-persistent-physical-adaptation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
            "resolved_experience_seed": experience_seed,
            "resolved_policy_perturbation_seed":
                policy_perturbation_seed,
            "resolved_context_proposal_seed":
                context_proposal_seed,
        },
        "training_signal":
            "three_candidate_persistent_physical_verified_horse_race",
        "memory_lifetime": (
            "one_decision_fresh_control"
            if args.reset_banks_each_round
            else "persistent_across_all_decisions"),
        "prefix_only": not complete_curriculum,
        "fast_adaptation_state": (
            "context_indexed_latent_strategy_memory"
            if args.strategy_memory_capacity > 0
            else "single_global_residual"),
        "semantic_or_utility_labels_used_for_training": False,
        "physical_reward_is_sovereign": True,
        "tensor_arena_role": "parity_audit_only",
        "trace": trace,
        "phase_summaries": phase_summaries,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "accounting": {
            "persistent_banks": args.banks,
            "physical_rounds": rounds,
            "candidate_save_reloads": expected_candidates,
            "state_save_reloads": expected_state,
            "total_replacements": total_replacements,
            "optimizer_updates": rounds,
            "candidate_verifier_bits":
                args.banks * args.bank_capacity * (
                    3 * rounds + strategy_candidate_evaluations),
            "strategy_candidate_evaluations":
                strategy_candidate_evaluations,
            "strategy_memory_save_reloads": strategy_save_reloads,
            "final_strategy_memory_count": (
                strategy_memory.count
                if strategy_memory is not None else 0),
            "strategy_memory_capacity": args.strategy_memory_capacity,
            "context_encoder_updates": context_updates,
            "context_encoder_mean_loss": (
                sum(context_losses) / len(context_losses)
                if context_losses else None),
            "context_encoder_mean_spsa_advantage": (
                sum(context_advantages) / len(context_advantages)
                if context_advantages else None),
            "context_encoder_scales": (
                context_encoder.log_scale.detach().exp().tolist()
                if context_encoder is not None else None),
            "unique_utility_contexts": len({
                tuple(summary["weights"])
                for summary in phase_summaries}),
            "soft_context_pair_count": soft_context_pair_count,
            "soft_context_action_divergent_pairs":
                soft_context_action_divergent_pairs,
            "soft_context_action_divergent_fraction": (
                soft_context_action_divergent_pairs
                / soft_context_pair_count
                if soft_context_pair_count else None),
            "soft_context_reward_divergent_pairs":
                soft_context_reward_divergent_pairs,
            "soft_context_reward_divergent_fraction": (
                soft_context_reward_divergent_pairs
                / soft_context_pair_count
                if soft_context_pair_count else None),
            "soft_context_mean_absolute_reward_delta": (
                soft_context_reward_delta_sum / soft_context_pair_count
                if soft_context_pair_count else None),
            "soft_context_direction_proposals_screened":
                soft_context_direction_proposals_screened,
            "soft_context_selected_preverifier_action_disagreements":
                soft_context_selected_preverifier_action_disagreements,
            "verifier_bits_per_reward_divergent_pair": (
                (
                    args.banks * args.bank_capacity * (
                        3 * rounds + strategy_candidate_evaluations)
                ) / soft_context_reward_divergent_pairs
                if soft_context_reward_divergent_pairs else None),
            "replayed_examples": 0,
            "requested_initial_histories_reproduced_exactly":
                requested_exact,
        },
        "maximum_physical_tensor_reward_difference":
            maximum_parity_difference,
        "maximum_cross_choice_regret": maximum_cross_choice_regret,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    if gate["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "gate": gate,
        "phase_summaries": phase_summaries,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
