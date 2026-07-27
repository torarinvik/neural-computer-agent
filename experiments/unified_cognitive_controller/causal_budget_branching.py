"""Generate leak-safe causal replay-budget decisions by branching trajectories.

Each row starts from one *identical* learner state and one identical sequence of
future sensory/reward episodes.  It then lets a low- and high-compute clone
learn from that future with different fixed replay budgets.  The verifier sees
the resulting capability and sample use; a future allocator sees only the
pre-branch learner summary.

This deliberately differs from a loss-based stopping probe.  Skipping or adding
updates changes every later state, so useful labels must be generated from whole
counterfactual continuations rather than local replay-loss deltas.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .audit_fifth_option_composition import load_fifth_router
from .audit_fourth_option_composition import load_router
from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_fifth_option_composition_race import (
    five_action_hierarchy,
    metrics,
    target_bits,
)
from .train_option_composition_race import OptionValueHead
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


STATE_FEATURE_PREFIX = (
    "log_observed_examples", "experience_step", "replay_loss",
    "prediction_mean", "prediction_std", "past_target_mean",
    "past_target_std", "residual_mean", "residual_std",
    "past_sign_agreement", "prediction_target_correlation",
    "gradient_norm", "prediction_q25", "prediction_q50", "prediction_q75",
    "target_q25", "target_q50", "target_q75",
)


def state_feature_names(latent_width: int) -> tuple[str, ...]:
    return (
        *STATE_FEATURE_PREFIX,
        *(f"latent_mean_{index}" for index in range(latent_width)),
        *(f"latent_std_{index}" for index in range(latent_width)),
    )


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    """Stable digest used only to prove both branches received the same input."""
    digest = hashlib.sha256()
    for tensor in tensors:
        cpu = tensor.detach().to("cpu").contiguous()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def state_digest(router: OptionValueHead) -> str:
    return tensor_digest([value for value in router.state_dict().values()])


def branch_state_digest(state: "BranchState") -> str:
    """Digest all causal state, including optimizer, replay, and RNG state."""
    digest = hashlib.sha256()
    digest.update(state_digest(state.router).encode())
    digest.update(tensor_digest([
        tensor for replay_row in state.replay for tensor in replay_row]).encode())
    digest.update(state.generator.get_state().detach().cpu().numpy().tobytes())
    optimizer_state = state.optimizer.state_dict()
    for parameter_id, values in sorted(optimizer_state["state"].items()):
        digest.update(str(parameter_id).encode())
        for key, value in sorted(values.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    for group in optimizer_state["param_groups"]:
        digest.update(repr(sorted(
            (key, value) for key, value in group.items() if key != "params")).encode())
    return digest.hexdigest()


def decision_features(
        router: OptionValueHead,
        replay: list[tuple[torch.Tensor, torch.Tensor]],
        *, step: int,
) -> list[float]:
    """Return only information available to the learner before it branches."""
    if not replay:
        raise ValueError("a branch needs at least one naturally observed batch")
    features = torch.cat([row[0] for row in replay])
    targets = torch.cat([row[1] for row in replay])
    router_width = int(router.network[0].normalized_shape[0])
    router_features = features[:, :router_width]
    prediction = router.q_values(router_features)
    difference = prediction[:, 1] - prediction[:, 0]
    residual = targets - difference
    loss = nn.functional.smooth_l1_loss(difference, targets)
    gradients = torch.autograd.grad(loss, router.parameters(), allow_unused=True)
    gradient_norm = math.sqrt(sum(
        float(gradient.detach().square().sum())
        for gradient in gradients if gradient is not None))
    centered_prediction = difference.detach() - difference.detach().mean()
    centered_target = targets - targets.mean()
    denominator = float(
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt())
    correlation = (
        float((centered_prediction * centered_target).sum() / denominator)
        if denominator > 0 else 0.0)
    prediction_quantiles = torch.quantile(
        difference.detach(), torch.tensor([0.25, 0.5, 0.75], device=features.device))
    target_quantiles = torch.quantile(
        targets, torch.tensor([0.25, 0.5, 0.75], device=features.device))
    feature_mean = features.mean(0)
    feature_std = features.std(0, unbiased=False)
    # These are all derived from past sensory/reward experience and internal
    # model state.  No verifier score, future stream, or branch outcome enters.
    return [
        math.log1p(float(features.shape[0])),
        float(step),
        float(loss),
        float(difference.mean()),
        float(difference.std(unbiased=False)),
        float(targets.mean()),
        float(targets.std(unbiased=False)),
        float(residual.mean()),
        float(residual.std(unbiased=False)),
        float((difference.sign() == targets.sign()).float().mean()),
        correlation,
        gradient_norm,
        *(float(value) for value in prediction_quantiles),
        *(float(value) for value in target_quantiles),
        *(float(value) for value in feature_mean),
        *(float(value) for value in feature_std),
    ]


def higher_budget_label(
        lower: dict[str, object], higher: dict[str, object], *,
        capability_tolerance: float,
) -> tuple[bool | None, dict[str, float | bool | None]]:
    """Allow extra compute only when it causally saves experience safely."""
    lower_bits = lower["stable_target_bits"]
    higher_bits = higher["stable_target_bits"]
    assert lower_bits is None or isinstance(lower_bits, int)
    assert higher_bits is None or isinstance(higher_bits, int)
    lower_utility = float(lower["final_utility"])
    higher_utility = float(higher["final_utility"])
    eligible_for_allocation = lower_bits is not None and higher_bits is not None
    saves_experience = bool(eligible_for_allocation and higher_bits < lower_bits)
    keeps_capability = higher_utility >= lower_utility - capability_tolerance
    # A pair that never reaches stable mastery has not taught us that low
    # compute is preferable. It has taught us that the current task/rung is
    # unsolved, and must never become a negative allocator training label.
    label = (
        bool(saves_experience and keeps_capability)
        if eligible_for_allocation else None)
    return label, {
        "lower_stable_bits": lower_bits,
        "higher_stable_bits": higher_bits,
        "lower_final_utility": lower_utility,
        "higher_final_utility": higher_utility,
        "eligible_for_allocation": eligible_for_allocation,
        "saves_experience": saves_experience,
        "keeps_capability": keeps_capability,
    }


@dataclass
class BranchState:
    router: OptionValueHead
    optimizer: torch.optim.Optimizer
    replay: list[tuple[torch.Tensor, torch.Tensor]]
    generator: torch.Generator

    def clone(self) -> "BranchState":
        router = copy.deepcopy(self.router)
        optimizer = torch.optim.AdamW(
            router.parameters(), lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=self.optimizer.param_groups[0]["weight_decay"])
        optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        generator = torch.Generator(device=self.generator.device)
        generator.set_state(self.generator.get_state())
        return BranchState(
            router=router,
            optimizer=optimizer,
            replay=[(features.detach().clone(), targets.detach().clone())
                    for features, targets in self.replay],
            generator=generator,
        )


def observe_and_replay(
        state: BranchState, features: torch.Tensor, targets: torch.Tensor,
        *, replay_updates: int, batch_size: int,
) -> None:
    state.replay.append((features.detach().clone(), targets.detach().clone()))
    replay_features = torch.cat([row[0] for row in state.replay])
    replay_targets = torch.cat([row[1] for row in state.replay])
    for _ in range(replay_updates):
        indices = torch.randint(
            0, replay_features.shape[0], (batch_size,),
            generator=state.generator, device=replay_features.device)
        router_width = int(state.router.network[0].normalized_shape[0])
        q_values = state.router.q_values(replay_features[indices, :router_width])
        prediction = q_values[:, 1] - q_values[:, 0]
        loss = nn.functional.smooth_l1_loss(prediction, replay_targets[indices])
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(state.router.parameters(), 1.0)
        state.optimizer.step()


def evaluate_branch(
        router: OptionValueHead, test_features: torch.Tensor,
        test_utilities: torch.Tensor, inherited_actions: torch.Tensor,
        *, new_action: int,
) -> dict[str, float]:
    with torch.no_grad():
        use_new = router(test_features[:, :router.network[0].normalized_shape[0]]).bool()
        actions = torch.where(
            use_new, torch.full_like(inherited_actions, new_action),
            inherited_actions)
        return metrics(actions, test_utilities, new_action)


def continue_branch(
        state: BranchState, episodes: list[tuple[torch.Tensor, torch.Tensor]],
        *, replay_updates: int, batch_size: int, test_features: torch.Tensor,
        test_utilities: torch.Tensor, inherited_actions: torch.Tensor,
        new_action: int, target_utility: float, bits_per_step: int,
) -> dict[str, object]:
    history = []
    for offset, (features, targets) in enumerate(episodes, start=1):
        observe_and_replay(
            state, features, targets, replay_updates=replay_updates,
            batch_size=batch_size)
        row = evaluate_branch(
            state.router, test_features, test_utilities, inherited_actions,
            new_action=new_action)
        row.update({
            "future_step": offset,
            "verifier_bits": offset * bits_per_step,
            "reaches_target": row["verified_utility"] >= target_utility,
        })
        history.append(row)
    return {
        "stable_target_bits": target_bits(history, stable=True),
        "final_utility": float(history[-1]["verified_utility"]),
        "history": history,
        "optimizer_updates": len(episodes) * replay_updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--three-option", type=Path, required=True)
    parser.add_argument("--four-router", type=Path, required=True)
    parser.add_argument("--fifth-router", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8280)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--branch-step", type=int, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument(
        "--write-threshold", type=float, default=0.5,
        help=("Memory-write regime for this whole logical stream. It is not "
              "provided to the allocator; it is visible only through the "
              "naturally produced latent evidence."))
    parser.add_argument(
        "--support-trials", type=int, default=1,
        help=("Number of naturally rendered demonstrations per lifetime. The "
              "allocator receives only their resulting latent evidence."))
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--prefix-replay-updates", type=int, default=48)
    parser.add_argument("--lower-replay-updates", type=int, default=48)
    parser.add_argument("--higher-replay-updates", type=int, default=56)
    parser.add_argument("--capability-tolerance", type=float, default=0.003)
    parser.add_argument("--practical-gain", type=float, default=0.02)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    branch_steps = sorted(set(args.branch_step))
    if not branch_steps or branch_steps[-1] >= args.steps or branch_steps[0] < 1:
        raise ValueError("branch steps must lie within the observed trajectory")
    if args.batch_size % args.capacity or args.test_contexts % args.capacity:
        raise ValueError("batch and test counts must divide by capacity")
    if min(args.prefix_replay_updates, args.lower_replay_updates,
           args.higher_replay_updates) < 1:
        raise ValueError("replay budgets must be positive")
    if args.higher_replay_updates <= args.lower_replay_updates:
        raise ValueError("higher replay budget must exceed lower budget")
    if not 0.0 <= args.write_threshold <= 1.0:
        raise ValueError("write threshold must lie in [0, 1]")
    if args.support_trials < 1:
        raise ValueError("support trials must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    champion = _load_head(args.champion_head, device)
    option3 = load_option(args.three_option, device)
    router4 = load_router(args.four_router, device)
    router5 = load_fifth_router(args.fifth_router, device)
    for module in (champion, option3, router4, router5):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    candidate_count, new_action = 6, 5
    router_input_width = int(router5.network[0].normalized_shape[0])
    costs = torch.arange(candidate_count, device=device, dtype=torch.float32) * 0.01
    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device,
        write_threshold=args.write_threshold, support_trials=args.support_trials,
        candidate_count=candidate_count, include_rank_features=True,
        include_latent_summary=True)
    test_utilities = test_outcomes - costs
    inherited_test_actions = five_action_hierarchy(
        router5, router4, option3, champion,
        test_features[:, :router_input_width])
    baseline_utility = metrics(
        inherited_test_actions, test_utilities, new_action)["verified_utility"]
    target_utility = baseline_utility + args.practical_gain

    episodes: list[tuple[torch.Tensor, torch.Tensor]] = []
    future_digest_tensors: list[torch.Tensor] = []
    for step in range(1, args.steps + 1):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold, candidate_count=candidate_count,
            support_trials=args.support_trials, include_latent_summary=True,
            include_rank_features=True)
        utilities = outcomes - costs
        inherited = five_action_hierarchy(
            router5, router4, option3, champion,
            features[:, :router_input_width])
        targets = utilities[:, new_action] - utilities.gather(
            1, inherited[:, None]).squeeze(1)
        episodes.append((features.detach(), targets.detach()))
        future_digest_tensors.extend((episodes[-1][0], episodes[-1][1]))
    full_episode_digest = tensor_digest(future_digest_tensors)

    router = OptionValueHead(router_input_width, 32).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed + 71_000_000)
    state = BranchState(router, optimizer, [], generator)
    examples = []
    for step, episode in enumerate(episodes, start=1):
        observe_and_replay(
            state, *episode, replay_updates=args.prefix_replay_updates,
            batch_size=args.batch_size)
        if step not in branch_steps:
            continue
        before_digest = branch_state_digest(state)
        before_features = decision_features(state.router, state.replay, step=step)
        lower_state, higher_state = state.clone(), state.clone()
        lower_initial_digest = branch_state_digest(lower_state)
        higher_initial_digest = branch_state_digest(higher_state)
        if lower_initial_digest != before_digest:
            raise RuntimeError("lower branch clone changed the decision state")
        if higher_initial_digest != before_digest:
            raise RuntimeError("higher branch clone changed the decision state")
        remaining = episodes[step:]
        remaining_digest = tensor_digest([
            tensor for episode_row in remaining for tensor in episode_row])
        lower = continue_branch(
            lower_state, remaining, replay_updates=args.lower_replay_updates,
            batch_size=args.batch_size, test_features=test_features,
            test_utilities=test_utilities, inherited_actions=inherited_test_actions,
            new_action=new_action, target_utility=target_utility,
            bits_per_step=args.batch_size * 2)
        higher = continue_branch(
            higher_state, remaining, replay_updates=args.higher_replay_updates,
            batch_size=args.batch_size, test_features=test_features,
            test_utilities=test_utilities, inherited_actions=inherited_test_actions,
            new_action=new_action, target_utility=target_utility,
            bits_per_step=args.batch_size * 2)
        label, outcome = higher_budget_label(
            lower, higher, capability_tolerance=args.capability_tolerance)
        examples.append({
            "branch_step": step,
            "state_features": before_features,
            "branch_state_digest": before_digest,
            "lower_initial_digest": lower_initial_digest,
            "higher_initial_digest": higher_initial_digest,
            "future_episode_digest": remaining_digest,
            "future_episode_count": len(remaining),
            "choose_higher_budget": label,
            "outcome": outcome,
            "lower": lower,
            "higher": higher,
        })

    report = {
        "schema": "causal-budget-branching-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "three_option": str(args.three_option),
            "four_router": str(args.four_router),
            "fifth_router": str(args.fifth_router),
            "report": str(args.report),
        },
        "feature_names": state_feature_names(int(episodes[0][0].shape[1])),
        "baseline_utility": baseline_utility,
        "target_utility": target_utility,
        "all_episode_digest": full_episode_digest,
        "examples": examples,
        "integrity": {
            "all_branches_started_from_identical_state": all(
                row["branch_state_digest"] == row["lower_initial_digest"]
                == row["higher_initial_digest"] for row in examples),
            "all_examples_have_future_episodes": all(
                row["future_episode_count"] > 0 for row in examples),
            "labels_are_verifier_only": True,
        },
        "accounting": {
            "base_prefix_optimizer_updates": args.steps * args.prefix_replay_updates,
            "branch_optimizer_updates": sum(
                int(row["lower"]["optimizer_updates"])
                + int(row["higher"]["optimizer_updates"])
                for row in examples),
            "unique_logical_lifetimes": args.steps * args.batch_size,
            "branch_count": len(examples),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
