"""Probe a k-action world, fit what explains it, and run the result.

The binary path gets its evidence through `episode_trace`, which installs a
program from the temporal bank, runs a lifetime, and inverts the reward. None
of that is needed here and most of it would be misleading: the probe's job is
to *choose what to try*, which under two actions was a question with no
content and under `k` is the whole difficulty.

So the probe policy is explicit and uniform over the action set. That is not
obviously the best policy -- it is the one whose cost can be reasoned about,
and the record measures what it costs rather than assuming it is free.

`run_choice_machine` steps a hypothesis directly, which is what the ceiling
measurement needs: it isolates what was learned from whether it compiles.
`run_choice_program_episode` runs the compiled artifact from a library record
instead, and that is what confirmation and admission use -- so nothing is kept
on the strength of a hypothesis that was never built.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .choice_induction import ChoiceTrace
from .counter_state_programs import nearest_cluster
from .rendered_environment import RenderedBrainWorkshopVerifier
from .rule_automata import RuleAutomaton

CHOICE_AGENT_SCHEMA = "neural-computer.choice-agent.v1"


def probe_episode(
    encoders,
    config,
    clusters: torch.Tensor,
    *,
    seed: int,
    policy_seed: int,
    action_count: int,
) -> ChoiceTrace:
    """One scored episode under a uniform-random policy.

    The stimulus seed and the policy seed are separate on purpose. Holding the
    world fixed while varying what the agent tries is the only way to attribute
    a difference in what it learned to the choosing rather than to the episode.
    """

    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    stream = config.streams[0]
    generator = torch.Generator().manual_seed(int(policy_seed))
    symbols: list[int] = []
    actions: list[int] = []
    rewards: list[int] = []
    eligible: list[bool] = []
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        if frame is None:
            raise ValueError("choice probe found no frame on the bound stream")
        with torch.no_grad():
            event = (
                encoders.vision(frame.unsqueeze(0))
                if stream == "vision"
                else encoders.audio(frame.unsqueeze(0))
            )
        symbols.append(int(nearest_cluster(event, clusters).item()))
        action = int(
            torch.randint(0, int(action_count), (1,), generator=generator).item()
        )
        actions.append(action)
        step = verifier.score(torch.tensor([action], dtype=torch.long))
        rewards.append(int(step.reward.item()))
        eligible.append(bool(step.eligible.item()))
    return ChoiceTrace(
        symbols=tuple(symbols),
        actions=tuple(actions),
        rewards=tuple(rewards),
        eligible=tuple(eligible),
        symbol_count=int(clusters.shape[0]),
        action_count=int(action_count),
    ).validate()


def fixed_policy_episode(
    encoders,
    config,
    clusters: torch.Tensor,
    *,
    seed: int,
    action: int,
    action_count: int,
) -> ChoiceTrace:
    """The control policy: always play the same thing.

    Under two actions this loses nothing -- the reward still names the target.
    Under `k` it is the policy that cannot learn, and how badly it fails is the
    measurement that shows exploration is doing work.
    """

    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    stream = config.streams[0]
    symbols: list[int] = []
    actions: list[int] = []
    rewards: list[int] = []
    eligible: list[bool] = []
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        with torch.no_grad():
            event = (
                encoders.vision(frame.unsqueeze(0))
                if stream == "vision"
                else encoders.audio(frame.unsqueeze(0))
            )
        symbols.append(int(nearest_cluster(event, clusters).item()))
        actions.append(int(action))
        step = verifier.score(torch.tensor([int(action)], dtype=torch.long))
        rewards.append(int(step.reward.item()))
        eligible.append(bool(step.eligible.item()))
    return ChoiceTrace(
        symbols=tuple(symbols),
        actions=tuple(actions),
        rewards=tuple(rewards),
        eligible=tuple(eligible),
        symbol_count=int(clusters.shape[0]),
        action_count=int(action_count),
    ).validate()


def run_choice_machine(
    machine: RuleAutomaton,
    encoders,
    config,
    clusters: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Drive an episode's answers from a hypothesis, and score them."""

    machine.validate()
    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    stream = config.streams[0]
    state = 0
    hits = scored = 0
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        with torch.no_grad():
            event = (
                encoders.vision(frame.unsqueeze(0))
                if stream == "vision"
                else encoders.audio(frame.unsqueeze(0))
            )
        symbol = int(nearest_cluster(event, clusters).item())
        action = int(machine.outputs[state][symbol])
        state = int(machine.transitions[state][symbol])
        step = verifier.score(torch.tensor([action], dtype=torch.long))
        if bool(step.eligible.item()):
            hits += int(step.reward.item())
            scored += 1
    return {"accuracy": hits / scored if scored else 0.0, "scored": scored}


def choice_config(base, rule, steps: int):
    """The same rendered task, carrying a rule that may have many actions."""

    return replace(base, steps=int(steps), rule=rule, match_rule="automaton").validate()


def run_choice_program_episode(record, encoders, config, clusters, *, seed: int):
    """Run a stored `k`-action program against the verifier.

    The record carries everything the executor needs -- program, start
    counters, alphabet, action count -- so confirmation runs the artifact that
    would be admitted rather than the hypothesis behind it.
    """

    from .choice_programs import run_choice_program

    return run_choice_program(
        record.program,
        encoders,
        config,
        clusters,
        action_count=record.action_count,
        seed=seed,
        initial_counters=record.initial_counters,
    )
