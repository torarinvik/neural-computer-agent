"""Choose the next program from feedback already paid for.

The accumulation curve costed the enumerating searcher at 120 episodes for a
rule it never solved. Each of those episodes returns a reward at every
eligible step -- 448 bits -- and the searcher reduces all of it to one
accuracy scalar and one accept/reject decision. The waste is not marginal.

It is also unnecessary, because the reward is *self-revealing*. The verifier
scores a press as correct or incorrect, so an episode's action sequence
together with its per-step rewards determines what the target wanted:

    target[t] = action[t]        if reward[t]
                1 - action[t]    otherwise

One episode with an arbitrary program therefore recovers the entire target
behaviour on that episode. Nothing here reads the rule, the automaton, or any
verifier state; it reads the feedback the agent is already given and declines
to throw it away. `test_feedback_proposer.py` checks the recovered target
against the generating rule to confirm the inversion is exact.

With `behaviour_signature` supplying each candidate's presses for free, the
selection problem becomes offline: rank every proposal by agreement with the
recovered target, and spend verifier evidence only on confirming the best
one. Search cost stops scaling with the size of the proposal list and starts
scaling with the number of hypotheses that actually need testing.

Two disciplines keep this honest.

The probe episode is *not* the evaluation episode. Ranking on the same
lifetime a winner is scored on would be fitting the test, so the probe runs
on one seed and the winner is confirmed on another. Agreement on the probe is
a selection signal; only the held-out episode is evidence.

And the ceiling is unchanged. Perfect ranking cannot gate a rule this program
family is unable to express, so a rule that stays unsolved under feedback
ranking is evidence about expressiveness rather than about search. Separating
those two has been the point of every ceiling in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .behaviour_signature import NO_ACTION, behaviour_signature, observe_stream

FEEDBACK_PROBE_SCHEMA = "neural-computer.feedback-probe.v1"


@dataclass(frozen=True)
class FeedbackProbe:
    """What one episode's per-step rewards say the target wanted."""

    target: tuple[int, ...]
    eligible: tuple[bool, ...]
    probe_label: str
    probe_accuracy: float

    def __post_init__(self) -> None:
        if len(self.target) != len(self.eligible):
            raise ValueError("probe target and eligibility must align")
        if not any(self.eligible):
            raise ValueError("a probe with no eligible step carries no information")

    @property
    def trials(self) -> int:
        return sum(1 for flag in self.eligible if flag)

    def agreement(self, signature: tuple[int, ...]) -> float:
        """Fraction of eligible steps where a candidate matches the target.

        This is exactly the eligible accuracy the candidate would have scored
        on the probe episode, computed without running it.
        """

        if len(signature) != len(self.target):
            raise ValueError("signature and probe cover different episodes")
        hits = sum(
            1
            for press, want, flag in zip(signature, self.target, self.eligible)
            if flag and press == want
        )
        return hits / self.trials

    def payload(self) -> dict[str, object]:
        return {
            "schema": FEEDBACK_PROBE_SCHEMA,
            "trials": self.trials,
            "probe_label": self.probe_label,
            "probe_accuracy": float(self.probe_accuracy),
            "target_press_rate": sum(
                1
                for want, flag in zip(self.target, self.eligible)
                if flag and want
            )
            / self.trials,
        }


def probe_target(lifetime, *, probe_label: str = "") -> FeedbackProbe:
    """Invert one scored lifetime into the target behaviour it was scoring.

    Takes the report a lifetime already returns, so a probe costs the episode
    it was going to cost anyway and nothing more.
    """

    actions = lifetime.actions.reshape(-1)
    rewards = lifetime.rewards.reshape(-1)
    present = lifetime.outcome_present.reshape(-1).bool()
    if actions.shape != rewards.shape or actions.shape != present.shape:
        raise ValueError("a probe needs aligned actions, rewards, and eligibility")
    recovered = torch.where(rewards.bool(), actions, 1 - actions)
    return FeedbackProbe(
        target=tuple(int(value) for value in recovered),
        eligible=tuple(bool(value) for value in present),
        probe_label=probe_label,
        probe_accuracy=float(lifetime.eligible_accuracy),
    )


def rank_by_agreement(
    signatures: dict[int, tuple[int, ...]],
    probe: FeedbackProbe,
) -> tuple[tuple[int, float], ...]:
    """Proposal indices best-first, with the agreement each would score.

    Ties keep proposal order, so the searcher's existing preference for
    simpler hypotheses survives ranking.
    """

    scored = [(index, probe.agreement(signature)) for index, signature in signatures.items()]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return tuple(scored)


def signatures_for(
    proposals,
    machine,
    bank,
    stream,
    *,
    install,
) -> dict[int, tuple[int, ...]]:
    """Every installable proposal's presses on a recorded stream.

    Costs no verifier evidence: the stream is an observation pass and the
    replay is offline.
    """

    signatures: dict[int, tuple[int, ...]] = {}
    for index, proposal in enumerate(proposals):
        if proposal.kind == "illegal_compose" or proposal.artifact is None:
            continue
        try:
            install(machine, bank, proposal)
        except (ValueError, RuntimeError):
            continue
        signature = behaviour_signature(machine, stream)
        if all(press == NO_ACTION for press in signature):
            continue
        signatures[index] = signature
    return signatures


def observation_stream(encoders, config, *, seed: int):
    """Re-exported so callers need one import for a feedback search."""

    return observe_stream(encoders, config, seed=seed)
