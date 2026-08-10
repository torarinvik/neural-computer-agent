"""Rendered replay-free acquisition of an external transition model.

This is a pressure test for the CPU/files boundary, not a claim that the
current random keypress decoder has mastered Brain Workshop.  The controller,
frontend, and decoder are frozen.  Fresh rendered verifier lifetimes produce
opaque planner-state transitions; an affine external transition bank consumes
each row once through sufficient statistics.  Recursive held-out error is
compared with a matched fresh bank, and the report keeps the claim narrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    AmodalEvent,
    ContentAddressedMemory,
    ControllerFeedback,
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    ExternalTransitionRollout,
    PolicyFreeAmodalRuntime,
)

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent

REPLAY_FREE_TRANSITION_AUDIT_SCHEMA = (
    "neural-computer.brainworkshop-replay-free-transition-acquisition-audit.v1"
)


@dataclass(frozen=True)
class ReplayFreeTransitionAcquisitionReport:
    schema: str
    status: str
    controller_unchanged: bool
    replay_free_bank: bool
    model_improved_on_heldout_rollout: bool
    trained_heldout_error: float
    fresh_heldout_error: float
    training_lifetimes: int
    total_logical_lifetimes: int
    unique_verifier_bits: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    external_slot_count: int
    external_sample_count: int
    fresh_sample_count: int
    schema_version: str = REPLAY_FREE_TRANSITION_AUDIT_SCHEMA

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _controller_digest(agent: CanonicalBrainWorkshopAgent) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(agent.controller.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _opaque_context(
    agent: CanonicalBrainWorkshopAgent,
    cue_symbol: int,
) -> torch.Tensor:
    """Derive one route key from an encoded rendered cue only."""

    encoded = agent.runtime.encoders["stimulus"](
        torch.tensor([cue_symbol], dtype=torch.long)
    )
    return F.normalize(encoded.detach(), dim=-1)[0]


def _run_lifetime(
    agent: CanonicalBrainWorkshopAgent,
    policy_free: PolicyFreeAmodalRuntime,
    bank: ExternalTransitionModelBank,
    context: torch.Tensor,
    *,
    n_back: int,
    steps: int,
    seed: int,
    cue_symbol: int,
    candidate_intentions: torch.Tensor,
    learn: bool,
) -> tuple[ExternalTransitionRollout, int]:
    """Run one fresh rendered lifetime and optionally consume its rows once."""

    verifier = NBackVerifier(
        batch_size=1,
        n_back=n_back,
        steps=steps,
        symbol_count=4,
        cue_symbol=cue_symbol,
        seed=seed,
    )
    verifier.reset()
    if isinstance(agent.runtime.memory, ContentAddressedMemory):
        agent.runtime.memory.clear()
    state = agent.initial_state(1, device=verifier.device)
    feedback = agent.initial_feedback(1, device=verifier.device)
    goal_state = torch.zeros(1, bank.state_width, device=verifier.device)
    previous = None
    outputs = []
    unique_verifier_bits = 0

    while not verifier.done:
        events = agent.runtime.encode_streams(
            {"stimulus": verifier.observation()}
        )
        output, next_state = policy_free.step_events(
            events,
            state,
            feedback,
            goal_state,
            candidate_intentions,
            horizon=1,
            beam_width=4,
        )
        if previous is not None and learn:
            observation = policy_free.transition_observation(previous, output)
            policy_free.learn_transition_once(observation, context)
        outputs.append(output)

        decision = agent.keypress_decoder.decide_from_logits(
            output.decoded["keypress"],
            sample=False,
        )
        scored = verifier.score(decision.key_index)
        unique_verifier_bits += int(scored.eligible.sum())
        feedback = ControllerFeedback(
            action=agent.keypress_encoder(decision.key_index),
            reward=scored.reward,
            propensity=decision.propensity,
            has_feedback=scored.eligible.to(scored.reward.dtype),
        )
        state = next_state
        previous = output

    if len(outputs) < 2:
        raise RuntimeError("transition audit lifetime needs at least two outputs")
    return (
        ExternalTransitionRollout(
            initial_state=outputs[0].state[0].detach().clone(),
            intentions=torch.cat(
                [output.intention.payload for output in outputs[:-1]], dim=0
            ).detach(),
            expected_states=torch.cat(
                [output.state for output in outputs[1:]], dim=0
            ).detach(),
        ).validate(
            state_width=bank.state_width,
            intention_width=bank.intention_width,
        ),
        unique_verifier_bits,
    )


def run_replay_free_transition_acquisition_audit(
    *,
    seed: int = 91,
    n_back: int = 2,
    steps: int = 6,
    training_lifetimes: int = 3,
    cue_symbol: int = 6,
) -> ReplayFreeTransitionAcquisitionReport:
    """Run the smallest rendered external-transition acquisition rung."""

    if min(n_back, steps, training_lifetimes) < 1:
        raise ValueError("transition audit dimensions must be positive")
    if cue_symbol < 4:
        raise ValueError("cue symbol must be outside the verifier vocabulary")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        event_width=4,
        intention_width=2,
        feedback_width=3,
        n_back=n_back,
        reader_kind="relation",
        seed=seed,
    )
    before = _controller_digest(agent)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)

    bank = ExternalTransitionModelBank(
        state_width=agent.controller.width * 3,
        intention_width=agent.controller.intention_width,
        context_width=agent.controller.width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    context = _opaque_context(agent, cue_symbol)
    bank.ensure_context(context)
    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(bank, beam_width=4),
    )
    generator = torch.Generator().manual_seed(seed + 7000)
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=generator,
    )

    training_bits = 0
    training_rows = 0
    for lifetime in range(training_lifetimes):
        _, bits = _run_lifetime(
            agent,
            policy_free,
            bank,
            context,
            n_back=n_back,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=cue_symbol,
            candidate_intentions=candidate_intentions,
            learn=True,
        )
        training_bits += bits
        training_rows += steps

    heldout, heldout_bits = _run_lifetime(
        agent,
        policy_free,
        bank,
        context,
        n_back=n_back,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=cue_symbol,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    trained_error = ExternalModelBasedPlanner(bank, beam_width=4).rollout_error(
        heldout,
        transition_context=context.unsqueeze(0),
    )

    fresh_bank = ExternalTransitionModelBank(
        state_width=bank.state_width,
        intention_width=bank.intention_width,
        context_width=bank.context_width,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    fresh_bank.ensure_context(context)
    fresh_error = ExternalModelBasedPlanner(
        fresh_bank,
        beam_width=4,
    ).rollout_error(
        heldout,
        transition_context=context.unsqueeze(0),
    )
    unchanged = before == _controller_digest(agent)
    improved = trained_error < fresh_error
    return ReplayFreeTransitionAcquisitionReport(
        schema=REPLAY_FREE_TRANSITION_AUDIT_SCHEMA,
        status=(
            "rendered_replay_free_transition_boundary"
            if unchanged and improved
            else "rendered_replay_free_transition_boundary_failed"
        ),
        controller_unchanged=unchanged,
        replay_free_bank=bank.replay_free_updates,
        model_improved_on_heldout_rollout=improved,
        trained_heldout_error=trained_error,
        fresh_heldout_error=fresh_error,
        training_lifetimes=training_lifetimes,
        total_logical_lifetimes=training_lifetimes + 1,
        unique_verifier_bits=training_bits + heldout_bits,
        transition_rows_consumed_once=training_rows,
        optimizer_updates=0,
        replayed_examples=0,
        external_slot_count=bank.context_count,
        external_sample_count=int(bank.models[0].sample_count),
        fresh_sample_count=int(fresh_bank.models[0].sample_count),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--training-lifetimes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    report = run_replay_free_transition_acquisition_audit(
        seed=args.seed,
        training_lifetimes=args.training_lifetimes,
        steps=args.steps,
    )
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report.payload(), indent=2) + "\n")
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    main()
