"""What a program *does*, computed without spending verifier evidence.

The accumulation curve found the searcher executing 120 proposals against the
verifier for a rule it could not solve, because proposals are enumerated in a
fixed order and every one costs an episode. Most of those proposals are not
distinct: they emit the same presses on the same stream and differ only in how
they were built.

Bottom-up enumerative synthesis has collapsed such candidates since Transit
(Udupa et al., 2013) and Escher (Albarghouthi et al., 2013): two programs are
*observationally equivalent* when they agree on every observed input, and only
one representative of each equivalence class is worth testing. This module is
that filter for this repository.

The signature is a program's action sequence on one encoded stimulus stream.
Computing it reads no reward. An observation pass drives the environment with
a constant action and discards the outcomes -- the same discipline
`prototype_templates.observe_events` already follows -- and every proposal is
then replayed against the recorded events offline. One pass prices the whole
proposal list instead of one proposal.

Dedup here is **lossless rather than approximate**, which is unusual and worth
the care it takes. Signatures are computed on the stream the proposals will
actually be scored on. Under a frozen controller with learning and sampling
off, a lifetime's actions are a deterministic function of the events and the
installed program, and eligible accuracy is a function of those actions and
the verifier's hidden labels. So equal signatures imply equal accuracy, and
skipping a duplicate cannot change any result -- it only declines to pay for
it twice. `tests/test_behaviour_signature.py` checks that against real runs
rather than taking the argument on faith.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neural_computer.interface import AmodalEventCollection

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
)

SIGNATURE_SCHEMA = "neural-computer.behaviour-signature.v1"
TICK_SECONDS = 0.01
# A tick that proposed nothing is still a distinguishable behaviour, and it
# must not collide with a real action index.
NO_ACTION = -1


def observe_stream(
    encoders: RenderedBrainWorkshopEncoders,
    config: RenderedBrainWorkshopConfig,
    *,
    seed: int,
    tick_seconds: float = TICK_SECONDS,
) -> tuple[AmodalEventCollection, ...]:
    """One observation pass, kept as the encoded events tick by tick.

    The pass emits a constant action and never reads the reward it produces,
    so it carries stimulus information and no verifier evidence. It costs one
    lifetime of environment time, which callers must account for exactly as
    they do for a template observation pass.
    """

    verifier = RenderedBrainWorkshopVerifier(config.validate(), seed=int(seed))
    collected: list[AmodalEventCollection] = []
    now = 0.0
    while not verifier.done:
        observation = verifier.observation()
        with torch.no_grad():
            collected.append(encoders.encode(observation, now=now))
        verifier.score(torch.zeros(1, dtype=torch.long))
        now += tick_seconds
    if not collected:
        raise ValueError("an observation pass produced no events")
    return tuple(collected)


def behaviour_signature(
    machine,
    stream: tuple[AmodalEventCollection, ...],
    *,
    tick_seconds: float = TICK_SECONDS,
) -> tuple[int, ...]:
    """Replay one installed program over recorded events; return its presses.

    History is cleared on both sides so a signature depends only on the
    program and the stream, never on whatever ran before it.
    """

    if not stream:
        raise ValueError("a behaviour signature needs a recorded stream")
    machine.reset_history()
    actions: list[int] = []
    now = 0.0
    with torch.no_grad():
        for events in stream:
            proposals = machine.tick(events, (), now=now, elapsed=tick_seconds)
            actions.append(
                int(proposals[0].action.item()) if proposals else NO_ACTION
            )
            now += tick_seconds
    machine.reset_history()
    return tuple(actions)


@dataclass(frozen=True)
class EquivalenceClasses:
    """Which proposals to execute, and which ones they stand in for."""

    representatives: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]
    signatures: tuple[tuple[int, ...] | None, ...]
    unsignable: tuple[int, ...]
    # Proposals the search would train before scoring. They are exempt from
    # collapsing and, downstream, from being ruled out on a signature that
    # predates their training.
    trained: tuple[int, ...] = ()

    @property
    def distinct(self) -> int:
        return len(self.representatives)

    def representative_of(self, index: int) -> int | None:
        """The proposal actually executed on behalf of `index`."""

        for leader, group in zip(self.representatives, self.members):
            if index in group:
                return leader
        return None

    def payload(self) -> dict[str, object]:
        return {
            "schema": SIGNATURE_SCHEMA,
            "distinct": self.distinct,
            "collapsed": sum(len(group) - 1 for group in self.members),
            "unsignable": len(self.unsignable),
        }


def learns_before_evaluation(proposal) -> bool:
    """Whether the search would run an acquire lifetime for this proposal.

    Mirrors the condition in `search_temporal_programs`. It matters here
    because such a proposal's behaviour *before* acquisition is not the
    behaviour it will be scored on -- an unacquired prototype is a zero row
    and presses accordingly. Signing it early would compare the wrong
    program, so these are never collapsed and never ruled out.
    """

    return proposal.kind in {"invent", "and"} and proposal.template is None


def partition_by_behaviour(
    proposals,
    machine,
    bank,
    stream: tuple[AmodalEventCollection, ...],
    *,
    install,
    tick_seconds: float = TICK_SECONDS,
) -> EquivalenceClasses:
    """Group proposals that press identically on the recorded stream.

    Order is preserved: the representative of a class is its earliest member,
    so the searcher's existing preference for simpler hypotheses is untouched
    and no previously recorded winner can be displaced by this filter.

    A proposal is *unsignable* when it cannot be installed, or when the search
    would train it before scoring it. Both are passed through rather than
    collapsed, so the search still sees and records them.
    """

    seen: dict[tuple[int, ...], int] = {}
    representatives: list[int] = []
    members: list[list[int]] = []
    signatures: list[tuple[int, ...] | None] = []
    unsignable: list[int] = []
    trained: list[int] = []
    for index, proposal in enumerate(proposals):
        if proposal.kind == "illegal_compose" or proposal.artifact is None:
            unsignable.append(index)
            continue
        if learns_before_evaluation(proposal):
            # Its own singleton class with no signature: never collapsed into
            # anything, and nothing collapsed into it.
            trained.append(index)
            representatives.append(index)
            members.append([index])
            signatures.append(None)
            continue
        try:
            install(machine, bank, proposal)
        except (ValueError, RuntimeError):
            unsignable.append(index)
            continue
        signature = behaviour_signature(machine, stream, tick_seconds=tick_seconds)
        leader = seen.get(signature)
        if leader is None:
            seen[signature] = index
            representatives.append(index)
            members.append([index])
            signatures.append(signature)
            continue
        members[representatives.index(leader)].append(index)
    return EquivalenceClasses(
        representatives=tuple(representatives),
        members=tuple(tuple(group) for group in members),
        signatures=tuple(signatures),
        unsignable=tuple(unsignable),
        trained=tuple(trained),
    )
