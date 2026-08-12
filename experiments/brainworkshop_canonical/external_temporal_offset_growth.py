"""Learn an external temporal address from scalar verifier outcomes.

The controller and learned event frontend remain frozen.  Each external file
contains a generic offset policy and a small learned readout.  The policy
chooses an opaque relative offset, the memory returns the corresponding event
token, and only the verifier's scalar outcome trains the chosen offset and
keypress output.  No n-back depth, target bit, correct action, or physical
memory address is exposed to the learner.
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
    ExternalTemporalHistoryMemory,
    ExternalTemporalOffsetSelector,
)

from .cross_family_rule_growth import RULES, CrossFamilyVerifier
from .runner import CanonicalBrainWorkshopAgent

TEMPORAL_OFFSET_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-offset-growth.v1"
)
EVENT_WIDTH = 16
INTENTION_WIDTH = 8
ACTION_COUNT = 2
MAX_OFFSET = 8
TARGET_CUE = 12
MASTERY_THRESHOLD = 0.80


@dataclass
class TemporalOffsetGrowthSystem:
    agent: CanonicalBrainWorkshopAgent


class ExternalTemporalCapabilityFile(nn.Module):
    """One opaque file that selects and consumes a temporal memory record."""

    def __init__(
        self,
        event_width: int = EVENT_WIDTH,
        *,
        max_offset: int = MAX_OFFSET,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(event_width, max_offset, hidden) < 1:
            raise ValueError("temporal capability dimensions must be positive")
        self.event_width = int(event_width)
        self.max_offset = int(max_offset)
        self.hidden = int(hidden)
        self.offset_selector = ExternalTemporalOffsetSelector(max_offset)
        self.readout = nn.Sequential(
            nn.Linear(event_width * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, ACTION_COUNT),
        )

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": TEMPORAL_OFFSET_GROWTH_SCHEMA,
            "event_width": self.event_width,
            "max_offset": self.max_offset,
            "hidden": self.hidden,
            "addressing": "learned_opaque_relative_offset_v1",
            "credit": "attempted_scalar_outcome_plus_offset_policy_credit_v1",
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def forward(
        self,
        event: torch.Tensor,
        history: ExternalTemporalHistoryMemory,
        scope: torch.Tensor,
        *,
        train: bool,
        forced_offset: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if event.ndim != 2 or event.shape[1] != self.event_width:
            raise ValueError("temporal capability event has the wrong shape")
        if forced_offset is None:
            offsets, log_probability, entropy = self.offset_selector(
                event.shape[0], sample=train
            )
        else:
            if not 1 <= forced_offset <= self.max_offset:
                raise ValueError("forced temporal offset is outside the file domain")
            offsets = torch.full(
                (event.shape[0],),
                forced_offset,
                dtype=torch.long,
                device=event.device,
            )
            log_probability = torch.zeros(
                event.shape[0], device=event.device, dtype=event.dtype
            )
            entropy = torch.zeros((), device=event.device, dtype=event.dtype)
        read = history.read_relative(offsets[:, None], scope=scope)
        retrieved = read.values[:, 0]
        present = read.present[:, 0].to(event.dtype).unsqueeze(-1)
        logits = self.readout(torch.cat((event, retrieved, present), dim=-1))
        return logits, offsets, log_probability, entropy


def _build(seed: int, *, file_count: int = 2) -> TemporalOffsetGrowthSystem:
    if file_count < 1:
        raise ValueError("temporal offset file count must be positive")
    torch.manual_seed(seed)
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=13,
        n_back=2,
        event_width=EVENT_WIDTH,
        intention_width=INTENTION_WIDTH,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
    )
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    return TemporalOffsetGrowthSystem(agent)


def _episode(
    system: TemporalOffsetGrowthSystem,
    file: ExternalTemporalCapabilityFile,
    *,
    family: str,
    batch_size: int,
    steps: int,
    seed: int,
    train: bool,
    entropy_weight: float,
    cue_symbol: int = TARGET_CUE,
    shuffle_outcomes: bool = False,
    forced_offset: int | None = None,
    reset_memory_each_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if family not in RULES:
        raise ValueError("unsupported temporal offset family")
    verifier = CrossFamilyVerifier(
        family=family,
        batch_size=batch_size,
        steps=steps,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    verifier.reset()
    agent = system.agent
    controller_state = agent.initial_state(batch_size, device="cpu")
    feedback = agent.initial_feedback(batch_size, device="cpu")
    scope = torch.arange(batch_size, dtype=torch.long)
    history = ExternalTemporalHistoryMemory(EVENT_WIDTH, scope_capacity=batch_size)
    selected_logits: list[torch.Tensor] = []
    offset_log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    delivered_rewards: list[torch.Tensor] = []
    eligible: list[torch.Tensor] = []
    selected_offsets: list[torch.Tensor] = []

    while not verifier.done:
        if forced_offset is None:
            offsets, offset_log_probability, entropy = file.offset_selector(
                batch_size,
                sample=train,
            )
        else:
            if not 1 <= forced_offset <= file.max_offset:
                raise ValueError("forced temporal offset is outside the file domain")
            offsets = torch.full(
                (batch_size,),
                forced_offset,
                dtype=torch.long,
            )
            offset_log_probability = torch.zeros(batch_size)
            entropy = torch.zeros(())
        if reset_memory_each_step:
            history.clear()
        # The capability file exposes logical lags starting at one.  The
        # canonical bridge reads before appending the current event, so the
        # external history query is one smaller and the current event remains
        # the persistent suffix of the controller input.
        bridge_offsets = offsets - 1
        with torch.no_grad():
            _controller_output, controller_state, bridge = (
                agent.runtime.step_streams_with_external_history(
                    {"stimulus": verifier.observation()},
                    controller_state,
                    feedback,
                    history,
                    bridge_offsets[:, None],
                    history_scope=scope,
                )
            )
        event = bridge.events.payload[:, -1].detach()
        retrieved = bridge.events.payload[:, 0]
        present = bridge.events.present[:, 0].to(event.dtype).unsqueeze(-1)
        logits = file.readout(torch.cat((event, retrieved, present), dim=-1))
        probabilities = logits.softmax(dim=-1)
        if train:
            action = torch.multinomial(probabilities, 1).squeeze(-1)
        else:
            action = logits.argmax(dim=-1)
        propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        scored = verifier.score(action)
        delivered_reward = (
            scored.reward.roll(1) if shuffle_outcomes else scored.reward
        )
        selected_logits.append(logits.gather(1, action[:, None]).squeeze(1))
        offset_log_probabilities.append(offset_log_probability)
        entropies.append(entropy)
        rewards.append(scored.reward)
        delivered_rewards.append(delivered_reward)
        eligible.append(scored.eligible)
        selected_offsets.append(offsets)
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(action),
            reward=delivered_reward,
            propensity=propensity,
            has_feedback=torch.ones(batch_size),
        )

    reward_tensor = torch.stack(rewards, dim=1)
    delivered_tensor = torch.stack(delivered_rewards, dim=1)
    eligible_tensor = torch.stack(eligible, dim=1)
    selected_tensor = torch.stack(selected_logits, dim=1)
    offset_log_tensor = torch.stack(offset_log_probabilities, dim=1)
    entropy_tensor = torch.stack(entropies)
    denominator = eligible_tensor.sum().clamp_min(1.0)
    accuracy = (reward_tensor * eligible_tensor).sum() / denominator
    if train:
        # Train only on the attempted action and its scalar verifier result.
        # The exact sampled propensity is still logged through feedback for
        # causal controls; no unattempted-action target enters this objective.
        action_loss = F.binary_cross_entropy_with_logits(
            selected_tensor[eligible_tensor], delivered_tensor[eligible_tensor]
        )
        offset_loss = -(
            (
                (delivered_tensor - 0.5).detach()
                * offset_log_tensor
                * eligible_tensor
            ).sum()
            / denominator
        )
        loss = action_loss + offset_loss
        if entropy_weight:
            loss = loss - entropy_weight * (
                (entropy_tensor[None, :] * eligible_tensor).sum() / denominator
            )
    else:
        loss = torch.zeros((), dtype=accuracy.dtype)
    return (
        loss,
        accuracy,
        int(eligible_tensor.sum().item()),
        torch.stack(selected_offsets, dim=1),
    )


def _train_file(
    system: TemporalOffsetGrowthSystem,
    file: ExternalTemporalCapabilityFile,
    *,
    family: str,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    learning_rate: float,
    entropy_weight: float,
    cue_symbol: int = TARGET_CUE,
    shuffle_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adam(file.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        torch.manual_seed(seed + update * 10_007)
        loss, accuracy, bits, _offsets = _episode(
            system,
            file,
            family=family,
            batch_size=batch_size,
            steps=steps,
            seed=seed + update,
            train=True,
            entropy_weight=entropy_weight,
            cue_symbol=cue_symbol,
            shuffle_outcomes=shuffle_outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(file.parameters(), max_norm=1.0)
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
    system: TemporalOffsetGrowthSystem,
    file: ExternalTemporalCapabilityFile,
    *,
    family: str,
    batch_size: int,
    steps: int,
    seed: int,
    lifetimes: int,
    cue_symbol: int = TARGET_CUE,
    forced_offset: int | None = None,
    reset_memory_each_step: bool = False,
) -> list[dict[str, float | int | list[int]]]:
    rows: list[dict[str, float | int | list[int]]] = []
    for lifetime in range(lifetimes):
        _loss, accuracy, bits, offsets = _episode(
            system,
            file,
            family=family,
            batch_size=batch_size,
            steps=steps,
            seed=seed + lifetime,
            train=False,
            entropy_weight=0.0,
            cue_symbol=cue_symbol,
            forced_offset=forced_offset,
            reset_memory_each_step=reset_memory_each_step,
        )
        eligible_offsets = offsets[:, RULES[family].warmup + 1 :]
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(accuracy),
                "mode_offset": int(torch.mode(eligible_offsets.flatten()).values),
                "unique_verifier_bits": batch_size * bits // max(batch_size, 1),
                "replayed_examples": 0,
            }
        )
    return rows


def _stable(rows: list[dict[str, float | int | list[int]]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _mean(rows: list[dict[str, float | int | list[int]]]) -> float:
    return sum(float(row["accuracy"]) for row in rows) / max(len(rows), 1)


def _digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.updates,
        args.batch_size,
        args.steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("temporal offset budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("temporal offset optimization parameters are invalid")
    if args.steps <= RULES["nback5"].warmup:
        raise ValueError("steps must include n-back-5 target trials")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    old_file = ExternalTemporalCapabilityFile()
    new_file = ExternalTemporalCapabilityFile()
    shuffled_control = ExternalTemporalCapabilityFile()

    old_history = _train_file(
        system,
        old_file,
        family="nback4",
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
    )
    old_before = _evaluate(
        system,
        old_file,
        family="nback4",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    old_digest_before = _digest(old_file)
    for parameter in old_file.parameters():
        parameter.requires_grad_(False)

    new_history = _train_file(
        system,
        new_file,
        family="nback5",
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
    )
    old_after = _evaluate(
        system,
        old_file,
        family="nback4",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    new_after = _evaluate(
        system,
        new_file,
        family="nback5",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 40_000,
        lifetimes=args.retention_lifetimes,
    )
    wrong_offset = 1
    wrong = _evaluate(
        system,
        new_file,
        family="nback5",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 50_000,
        lifetimes=args.retention_lifetimes,
        forced_offset=wrong_offset,
    )
    missing = _evaluate(
        system,
        new_file,
        family="nback5",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 60_000,
        lifetimes=args.retention_lifetimes,
        reset_memory_each_step=True,
    )
    _train_file(
        system,
        shuffled_control,
        family="nback5",
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 70_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        shuffle_outcomes=True,
    )
    shuffled = _evaluate(
        system,
        shuffled_control,
        family="nback5",
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    new_modes = [int(row["mode_offset"]) for row in new_after]
    gates = {
        "old_file_mastered_before_growth": _stable(old_before),
        "new_file_mastered_after_growth": _stable(new_after),
        "old_file_retained_after_growth": _stable(old_after),
        "old_file_unchanged": old_digest_before == _digest(old_file),
        "offset_policy_prefers_nback5": min(new_modes) == 5,
        "wrong_offset_rejects_mastery": _mean(wrong) < 0.70,
        "missing_history_rejects_mastery": _mean(missing) < MASTERY_THRESHOLD,
        "shuffled_outcome_rejects_mastery": _mean(shuffled) < MASTERY_THRESHOLD,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    primary_bits = args.batch_size * args.updates * (
        (args.steps - RULES["nback4"].warmup)
        + (args.steps - RULES["nback5"].warmup)
    )
    control_bits = args.batch_size * args.updates * (
        args.steps - RULES["nback5"].warmup
    )
    audit_rows = (*old_before, *old_after, *new_after, *wrong, *missing, *shuffled)
    report = {
        "schema": TEMPORAL_OFFSET_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-only discovery of a reusable external relative temporal "
            "offset with protected-file retention; not arbitrary addressing, "
            "unrestricted memory growth, or general continual learning."
        ),
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "memory": "external_temporal_history_memory_v1",
            "file": "external_temporal_capability_file_v1",
            "history_transport": "canonical_runtime_external_history_event_bridge_v2",
            "history_causality": "read_before_current_append",
            "history_persistence": "current_tokens_only_transient_prior_context",
            "bridge_offset_semantics": "logical_lag_minus_one_for_pre_append_relative_read",
            "address_policy": "external_temporal_offset_selector_v1",
            "credit": "attempted_bce_output_plus_scalar_offset_policy_credit",
            "max_offset": MAX_OFFSET,
            "source_family": "nback4",
            "target_family": "nback5",
            "target_cue": TARGET_CUE,
        },
        "seed": args.seed,
        "old_history_tail": old_history[-5:],
        "new_history_tail": new_history[-5:],
        "evaluation": {
            "old_before": old_before,
            "old_after": old_after,
            "new_after": new_after,
            "wrong_offset": wrong,
            "missing_history": missing,
            "shuffled_control": shuffled,
            "new_offset_modes": new_modes,
            "offset_probabilities": new_file.offset_selector.logits.softmax(-1)
            .detach()
            .tolist(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": primary_bits,
            "control_verifier_bits": control_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"]) for row in audit_rows
            ),
            "unique_logical_lifetimes": args.batch_size
            * args.updates
            * 2,
            "control_logical_lifetimes": args.batch_size * args.updates,
            "optimizer_updates": args.updates * 2,
            "control_optimizer_updates": args.updates,
            "route_memory_updates": 0,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": primary_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_temporal_offset_growth"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
