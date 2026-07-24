from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.syllogimous_latent_agent.data import collate_episodes

from .consolidation import (LearnedConsolidator, ReplayScore, score_sensory_replay,
                            transactional_consolidate_many)
from .lifetime import generate_sensory_lifetime
from .memory import PersistentMemory
from .model import NeuralComputerAgent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_controller(path: Path, device: torch.device) -> NeuralComputerAgent:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("base_config", payload["arguments"])
    read_top_k = int(payload.get("arguments", {}).get("read_top_k", 4))
    model = NeuralComputerAgent(config["hidden"], config["workspace_slots"], config["heads"],
                                config["thought_steps"], config["choices"], read_top_k).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
        raise ValueError(f"incompatible controller checkpoint: {incompatible}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def build_stream(model: NeuralComputerAgent, device: torch.device, seed: int,
                 contexts: int, delay: int, choices: int, threshold: float):
    memory = PersistentMemory.empty(max(4, contexts * (delay + 1)), model.hidden,
                                    device=device, growth_chunk=16)
    queries, audit_queries = [], []
    for context in range(contexts):
        lifetime = generate_sensory_lifetime(
            seed + context, associations=1, delay=delay, choices=choices,
            heldout=True, contextual=True, audit_variants=1)
        queries.append(lifetime.episodes[-1])
        audit_queries.extend(lifetime.audit_queries)
        for episode in lifetime.episodes[:-1]:
            batch = collate_episodes([episode])
            output = model(batch["frames"].to(device), batch["pcm"].to(device),
                           batch["mask"].to(device), memory)
            admitted = output.write_strengths >= threshold
            if admitted.any():
                memory.write(output.write_keys[admitted], output.write_values[admitted],
                             output.write_strengths[admitted], threshold=0.0)
    return memory, queries, audit_queries


def verifier(model, episodes, device):
    if not episodes:
        return lambda memory: ReplayScore(0, 0)
    return lambda memory: score_sensory_replay(model, memory, episodes, device)


def consolidate_stream(policy: LearnedConsolidator, optimizer, memory, verification_episodes,
                       model, device, attempts: int, *, training: bool,
                       loss_tolerance: float | None = None, rehearsal_groups: int = 2,
                       grouping_seed: int = 0, reward_episodes=None,
                       generalization_reward_weight: float = 0.0,
                       autonomous_stop: bool = False,
                       trajectory_stop: bool = False,
                       stop_threshold: float = 0.5):
    current = memory
    accepted = 0
    rewards = []
    stopped = False
    if not 1 <= rehearsal_groups <= len(verification_episodes):
        raise ValueError("rehearsal_groups must fit the verification episode count")
    for attempt in range(attempts):
        proposal = policy.sample(current) if training else policy.propose(current)
        if proposal is None:
            break
        should_stop = (float(torch.sigmoid(policy.stop_logit(current)).detach()) >=
                       stop_threshold
                       if trajectory_stop else
                       float(torch.sigmoid(
                           policy.rewrite_logit(current, proposal)).detach()) < 0.5)
        if autonomous_stop and should_stop:
            stopped = True
            break
        generator = random.Random(grouping_seed + attempt * 1_000_003)
        order = list(verification_episodes)
        generator.shuffle(order)
        groups = [order[index::rehearsal_groups] for index in range(rehearsal_groups)]
        checks = [verifier(model, group, device) for group in groups]
        result = transactional_consolidate_many(
            current, proposal, checks, storage_reward=0.01, error_penalty=2.0,
            loss_tolerance=loss_tolerance)
        reward = result.reward if result.committed else min(result.reward, -0.002)
        if result.committed and reward_episodes and generalization_reward_weight:
            reward_check = verifier(model, reward_episodes, device)
            reward_before = reward_check(current)
            reward_after = reward_check(result.memory)
            generalization_gain = (reward_after.accuracy - reward_before.accuracy +
                                   0.1 * (reward_before.loss - reward_after.loss))
            reward += generalization_reward_weight * generalization_gain
        rewards.append(reward)
        if training and proposal.log_probability is not None:
            optimizer.zero_grad(set_to_none=True)
            (-proposal.log_probability * reward).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if result.committed:
            current = result.memory
            accepted += 1
    verifier_queries = len(verification_episodes)
    # Every checked query is scored before and after each attempted transaction.
    verifier_queries *= 2 * len(rewards)
    return current, accepted, sum(rewards) / max(1, len(rewards)), verifier_queries, stopped


def tournament_consolidate_stream(
        policy: LearnedConsolidator, optimizer, memory, verification_episodes,
        reward_episodes, model, device, attempts: int, candidates: int, *,
        rehearsal_groups: int = 1, grouping_seed: int = 0,
        autonomous_stop: bool = False, storage_value: float = 0.01,
        stop_loss_weight: float = 4.0, trajectory_stop: bool = False):
    """Distill the best verified latent rewrite from a sampled candidate set.

    Candidate generation sees only controller-created memory latents. Public
    sensory outcomes rank candidates during training but never become model inputs.
    """
    if candidates < 2:
        raise ValueError("a tournament requires at least two candidates")
    current = memory
    accepted = 0
    gains = []
    verifier_queries = 0
    stopped = False
    trajectory = []
    initial_rows = memory.count
    for attempt in range(attempts):
        generator = random.Random(grouping_seed + attempt * 1_000_003)
        order = list(verification_episodes)
        generator.shuffle(order)
        groups = [order[index::rehearsal_groups] for index in range(rehearsal_groups)]
        checks = [verifier(model, group, device) for group in groups]
        reward_check = verifier(model, reward_episodes, device)
        reward_before = reward_check(current)
        if trajectory_stop:
            trajectory.append((current, reward_before))
        contenders = []
        decision_examples = []
        stop_rank = (reward_before.correct, -reward_before.loss, 0)
        for _ in range(candidates):
            proposal = policy.sample(current)
            if proposal is None:
                break
            result = transactional_consolidate_many(
                current, proposal, checks, storage_reward=0.01, error_penalty=2.0)
            verifier_queries += 2 * len(verification_episodes)
            if not result.committed:
                decision_examples.append((proposal, 0.0))
                continue
            reward_after = reward_check(result.memory)
            verifier_queries += len(reward_episodes)
            # Lexicographic correctness first; confidence breaks equal-accuracy ties.
            rank = (reward_after.correct, -reward_after.loss + storage_value)
            gain = (reward_after.accuracy - reward_before.accuracy +
                    0.1 * (reward_before.loss - reward_after.loss))
            contenders.append((rank, gain, proposal, result.memory))
            decision_examples.append((proposal, float(rank + (1,) > stop_rank)))
        ranked = [(rank + (1,), gain, proposal, candidate_memory)
                  for rank, gain, proposal, candidate_memory in contenders]
        best_candidate = max(ranked, key=lambda item: item[0]) if ranked else None
        if autonomous_stop and not trajectory_stop:
            ranked.append((stop_rank, 0.0, None, current))
        if not ranked:
            gains.append(0.0)
            continue
        _, gain, winner, winner_memory = max(ranked, key=lambda item: item[0])
        optimizer.zero_grad(set_to_none=True)
        loss = current.keys.new_zeros(())
        if autonomous_stop and not trajectory_stop and decision_examples:
            decision_losses = []
            for decision_proposal, target in decision_examples:
                rewrite_logit = policy.rewrite_logit(current, decision_proposal)
                rewrite_target = rewrite_logit.new_tensor(target)
                item_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    rewrite_logit, rewrite_target)
                if target == 0.0:
                    item_loss = item_loss * stop_loss_weight
                decision_losses.append(item_loss)
            loss = torch.stack(decision_losses).mean()
        if winner is not None:
            if winner.log_probability is None:
                raise RuntimeError("sampled tournament proposal lacks a log probability")
            loss = loss - winner.log_probability
        elif best_candidate is None:
            gains.append(0.0)
            stopped = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if winner is None:
            stopped = True
            gains.append(0.0)
            break
        current = winner_memory
        accepted += 1
        gains.append(gain)
    if trajectory_stop:
        final_score = verifier(model, reward_episodes, device)(current)
        trajectory.append((current, final_score))
        # A state should STOP unless a later state on the completed rollout has
        # strictly better correctness-first utility. Sensory outcomes construct
        # targets during training; the deployed head receives latent memory only.
        ranks = [(score.correct,
                  -score.loss + storage_value * (initial_rows - state.count))
                 for state, score in trajectory]
        labels = []
        for index, rank in enumerate(ranks):
            future = max(ranks[index + 1:], default=rank)
            labels.append(float(future <= rank))
        optimizer.zero_grad(set_to_none=True)
        stop_losses = []
        for (state, _), target in zip(trajectory, labels):
            logit = policy.stop_logit(state)
            item = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, logit.new_tensor(target))
            if target == 1.0:
                item = item * stop_loss_weight
            stop_losses.append(item)
        torch.stack(stop_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.stop_head.parameters(), 1.0)
        optimizer.step()
    return current, accepted, sum(gains) / max(1, len(gains)), verifier_queries, stopped


def split_queries(queries):
    """Replay/commit partitions guide transactions; audit is never consulted."""
    if len(queries) < 4:
        midpoint = max(1, len(queries) // 2)
        return queries[:midpoint], queries[midpoint:], queries
    audit_count = max(1, len(queries) // 4)
    verified = queries[:-audit_count]
    midpoint = max(1, len(verified) // 2)
    return verified[:midpoint], verified[midpoint:], queries[-audit_count:]


@torch.no_grad()
def evaluate(policy, model, device, args, *, streams: int, seed: int):
    totals = {"append_accuracy": 0.0, "consolidated_accuracy": 0.0,
              "append_audit_accuracy": 0.0, "consolidated_audit_accuracy": 0.0,
              "append_loss": 0.0, "consolidated_loss": 0.0,
              "append_audit_loss": 0.0, "consolidated_audit_loss": 0.0,
              "append_rows": 0.0, "consolidated_rows": 0.0,
              "accepted_rewrites": 0.0, "verifier_queries": 0.0,
              "stop_rate": 0.0, "stopped_rows": 0.0}
    for index in range(streams):
        memory, queries, audit = build_stream(
            model, device, seed + index * args.contexts, args.contexts,
            args.delay, args.choices, args.threshold)
        verification = queries
        baseline = score_sensory_replay(model, memory, queries, device)
        baseline_audit = score_sensory_replay(model, memory, audit, device)
        consolidated, accepted, _, verifier_queries, stopped = consolidate_stream(
            policy, None, memory, verification, model, device, args.attempts, training=False,
            loss_tolerance=args.loss_tolerance, rehearsal_groups=args.rehearsal_groups,
            grouping_seed=seed + index * 97, autonomous_stop=args.autonomous_stop,
            trajectory_stop=args.trajectory_stop, stop_threshold=args.stop_threshold)
        final = score_sensory_replay(model, consolidated, queries, device)
        final_audit = score_sensory_replay(model, consolidated, audit, device)
        totals["append_accuracy"] += baseline.accuracy
        totals["consolidated_accuracy"] += final.accuracy
        totals["append_audit_accuracy"] += baseline_audit.accuracy
        totals["consolidated_audit_accuracy"] += final_audit.accuracy
        totals["append_loss"] += baseline.loss
        totals["consolidated_loss"] += final.loss
        totals["append_audit_loss"] += baseline_audit.loss
        totals["consolidated_audit_loss"] += final_audit.loss
        totals["append_rows"] += memory.count
        totals["consolidated_rows"] += consolidated.count
        totals["accepted_rewrites"] += accepted
        totals["verifier_queries"] += verifier_queries
        totals["stop_rate"] += float(stopped)
        totals["stopped_rows"] += consolidated.count if stopped else 0
    result = {key: value / streams for key, value in totals.items()}
    result["lookup_reduction"] = 1.0 - result["consolidated_rows"] / max(1.0, result["append_rows"])
    return result


@torch.no_grad()
def calibrate_stop_threshold(policy, model, device, args, *, streams: int, seed: int):
    """Choose an accuracy-safe operating point on training-only sensory streams."""
    forced_args = copy.copy(args)
    forced_args.autonomous_stop = False
    forced = evaluate(policy, model, device, forced_args, streams=streams, seed=seed)
    candidates = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                  0.90, 0.95, 1.01)
    trials = []
    selected = candidates[-1]
    for threshold in candidates:
        candidate_args = copy.copy(args)
        candidate_args.stop_threshold = threshold
        result = evaluate(policy, model, device, candidate_args,
                          streams=streams, seed=seed)
        useful = result["verifier_queries"] < forced["verifier_queries"]
        safe = (useful and
                result["consolidated_accuracy"] >= forced["consolidated_accuracy"] and
                result["consolidated_audit_accuracy"] >=
                forced["consolidated_audit_accuracy"] and
                result["consolidated_loss"] <= forced["consolidated_loss"] and
                result["consolidated_audit_loss"] <=
                forced["consolidated_audit_loss"])
        trials.append({"threshold": threshold, "safe": safe, "useful": useful,
                       "accuracy": result["consolidated_accuracy"],
                       "audit_accuracy": result["consolidated_audit_accuracy"],
                       "loss": result["consolidated_loss"],
                       "audit_loss": result["consolidated_audit_loss"],
                       "verifier_queries": result["verifier_queries"]})
        if safe:
            selected = threshold
            break
        # Raising the threshold can only remove STOP decisions. If the current
        # cutoff never stops, no more conservative cutoff can save work.
        if not useful:
            break
    return selected, {"forced": forced, "trials": trials}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transactional latent-memory consolidation")
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-streams", type=int, default=128)
    parser.add_argument("--eval-streams", type=int, default=64)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument("--loss-tolerance", type=float,
                        help="also reject rewrites whose replay loss rises by more than this")
    parser.add_argument("--rehearsal-groups", type=int, default=2,
                        help="independently preserve this many compute-matched replay groups")
    parser.add_argument("--generalization-reward-weight", type=float, default=0.0,
                        help="policy reward for improvement on training-only sensory variants")
    parser.add_argument("--tournament-candidates", type=int, default=1,
                        help="sample this many rewrites and distill the best safe candidate")
    parser.add_argument("--autonomous-stop", action="store_true",
                        help="learn and use a latent decision to stop consolidation")
    parser.add_argument("--trajectory-stop", action="store_true",
                        help="train STOP from complete rollout returns instead of myopic rewrites")
    parser.add_argument("--autonomous-storage-value", type=float, default=0.01,
                        help="small confidence-equivalent value of removing one memory row")
    parser.add_argument("--stop-loss-weight", type=float, default=4.0,
                        help="balance the single STOP label against preceding rewrite labels")
    parser.add_argument("--stop-threshold", type=float, default=0.5,
                        help="deployment probability threshold for the latent STOP decision")
    parser.add_argument("--calibration-streams", type=int, default=0,
                        help="training-only streams used to select an accuracy-safe STOP threshold")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_controller(args.controller, device)
    policy = LearnedConsolidator(model.hidden, args.hidden).to(device)
    if args.initial_policy is not None:
        policy_payload = torch.load(args.initial_policy, map_location=device, weights_only=False)
        policy.load_state_dict(policy_payload["policy"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for index in range(args.train_streams):
        memory, queries, reward_queries = build_stream(
            model, device, index * args.contexts, args.contexts,
            args.delay, args.choices, args.threshold)
        if args.tournament_candidates > 1:
            final, accepted, reward, verifier_queries, stopped = tournament_consolidate_stream(
                policy, optimizer, memory, queries, reward_queries, model, device,
                args.attempts, args.tournament_candidates,
                rehearsal_groups=args.rehearsal_groups,
                grouping_seed=args.seed * 10_000 + index,
                autonomous_stop=args.autonomous_stop,
                storage_value=args.autonomous_storage_value,
                stop_loss_weight=args.stop_loss_weight,
                trajectory_stop=args.trajectory_stop)
        else:
            final, accepted, reward, verifier_queries, stopped = consolidate_stream(
                policy, optimizer, memory, queries, model, device,
                args.attempts, training=True, loss_tolerance=args.loss_tolerance,
                rehearsal_groups=args.rehearsal_groups,
                grouping_seed=args.seed * 10_000 + index,
                reward_episodes=reward_queries,
                generalization_reward_weight=args.generalization_reward_weight,
                autonomous_stop=args.autonomous_stop,
                trajectory_stop=args.trajectory_stop,
                stop_threshold=args.stop_threshold)
        if (index + 1) % max(1, args.train_streams // 8) == 0:
            row = {"stream": index + 1, "reward": reward, "accepted": accepted,
                   "rows_before": memory.count, "rows_after": final.count,
                   "stopped": stopped}
            row["verifier_queries"] = verifier_queries
            history.append(row)
            print(json.dumps(row), flush=True)
    calibration = None
    if args.calibration_streams:
        if not (args.autonomous_stop and args.trajectory_stop):
            raise ValueError("STOP calibration requires --autonomous-stop --trajectory-stop")
        args.stop_threshold, calibration = calibrate_stop_threshold(
            policy, model, device, args, streams=args.calibration_streams,
            seed=700_000 + args.seed * 10_000)
    evaluation = evaluate(policy, model, device, args, streams=args.eval_streams,
                          seed=1_000_000 + args.seed * 10_000)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "width": model.hidden,
                "arguments": vars(args)}, args.checkpoint)
    report = {"schema": "syllogimous-transactional-consolidation-v1",
              "controller_weights_frozen": True, "sensory_replay_only": True,
              "history": history, "evaluation": evaluation,
              "calibration": calibration,
              "training_seconds": time.perf_counter() - started,
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation": evaluation}), flush=True)


if __name__ == "__main__":
    main()
