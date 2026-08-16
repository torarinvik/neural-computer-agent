"""Wanting to see something new, written as a task rather than as a bonus.

The probe policy has been uniform since navigation started, and the records
have called that out every time without doing anything about it. Uniform
wandering covers a small world eventually and a larger one never.

Curiosity-driven learning (Burda et al., 2018) is the standard answer: reward
the agent for surprise, and it explores fifty-four games with no score at all.
Agent57 (Badia et al., 2020) sharpens it into two timescales -- novelty *within*
an episode, modulated by novelty *across* the agent's whole life -- and runs a
family of policies at different horizons rather than one.

Both are deep-RL systems and neither transplants directly. What transplants is
the shape, and it lands somewhere convenient: **novelty is a weight vector over
the cumulants we already have.**

That is worth stating plainly, because it is the argument for having built
successor features first. Novelty is a *non-stationary* task -- every step you
take makes the place you are standing on less interesting -- and a
non-stationary task is exactly the thing successor features make cheap. The
occupancies do not change when the task does. Re-aiming the agent at whatever
is now most novel costs one dot product per stored policy, not a replan.

So there is no intrinsic reward channel here, no bonus added to a return, and
no second value function. There is a `w` that changes every step.

**The two timescales, kept apart.** Episodic novelty is what has not been seen
*this* episode and resets; lifelong novelty is what has rarely been seen at
all, and decays over the run. Agent57 combines them multiplicatively, with the
lifelong term clipped so it modulates rather than dominates, and the same is
done here.

**What counts as novel is the load-bearing choice.** Prediction-error curiosity
has a famous failure -- the noisy television, an inexhaustible source of
surprise that the agent can sit and watch forever. It is not hypothetical here:
the distractor from the identity work is a noisy television. `scene` novelty
counts whole readings, and with something else moving in the frame no reading
ever repeats, so everything looks equally novel and the signal carries no
direction at all. `own` novelty counts only the part of the scene the agent
was measured to control. Both are implemented, because the second is only
worth its complexity if the first actually fails.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

NOVELTY_SCHEMA = "neural-computer.novelty.v1"

# Agent57 clips the lifelong term into [1, L] so that it modulates the episodic
# signal rather than replacing it. Same reason here: a place the agent has
# never visited in its life should be *more* attractive than one it has, but
# not so much more that within-episode structure stops mattering.
LIFELONG_CLIP = 5.0


@dataclass
class NoveltyCounts:
    """Two timescales of "have I seen this", kept separate on purpose.

    `episodic` resets between episodes and is what stops the agent pacing
    between two places. `lifelong` never resets and is what stops it
    re-exploring the same corner of the world every episode. Agent57 keeps
    both because either alone is defeated by a different failure.
    """

    alphabet: int
    episodic: dict[int, int] = field(default_factory=dict)
    lifelong: dict[int, int] = field(default_factory=dict)
    # Readings rather than places: the ungated view, kept so the noisy
    # television can be measured rather than asserted.
    readings: dict[tuple[int, ...], int] = field(default_factory=dict)
    readings_at: dict[int, dict[tuple[int, ...], int]] = field(default_factory=dict)

    def start_episode(self) -> None:
        self.episodic = {}

    def observe(self, place: int, reading: Sequence[int]) -> None:
        place = int(place)
        self.episodic[place] = self.episodic.get(place, 0) + 1
        self.lifelong[place] = self.lifelong.get(place, 0) + 1
        key = tuple(sorted(int(symbol) for symbol in reading))
        self.readings[key] = self.readings.get(key, 0) + 1
        here = self.readings_at.setdefault(place, {})
        here[key] = here.get(key, 0) + 1

    def reading_novelty(self, reading: Sequence[int]) -> float:
        key = tuple(sorted(int(symbol) for symbol in reading))
        return 1.0 / math.sqrt(1 + self.readings.get(key, 0))

    def weights(self, *, gated: bool = True) -> torch.Tensor:
        """The task vector: how much it is worth being at each place, now.

        Handed straight to generalised policy improvement as `w`. Nothing is
        recomputed to produce it -- the stored occupancies are unchanged and
        only the thing they are dotted with has moved.
        """

        if not gated:
            # Novelty as a property of the whole reading, charged to the place
            # the agent was standing on when it saw it. Nothing here decides in
            # advance that this fails -- it is computed the same way in both
            # conditions, and what it does is measured.
            #
            # With nothing else moving, the reading is essentially the agent's
            # own place plus a goal that is fixed for the episode, so reading
            # counts track place counts and this closely follows the gated
            # vector. With a distractor, no reading ever repeats, every count
            # stays at one, and the vector goes flat -- which is what having no
            # signal looks like from the inside.
            weights = torch.ones(self.alphabet, dtype=torch.float64)
            for place in range(self.alphabet):
                here = self.readings_at.get(place)
                if not here:
                    continue
                total = sum(here.values())
                weights[place] = (
                    sum(
                        count / math.sqrt(1 + self.readings.get(key, 0))
                        for key, count in here.items()
                    )
                    / total
                )
            return weights

        seen = [self.lifelong.get(place, 0) for place in range(self.alphabet)]
        average = sum(seen) / max(1, len(seen))
        weights = torch.zeros(self.alphabet, dtype=torch.float64)
        for place in range(self.alphabet):
            episodic = 1.0 / math.sqrt(1 + self.episodic.get(place, 0))
            modulator = math.sqrt((1.0 + average) / (1.0 + seen[place]))
            weights[place] = episodic * min(max(modulator, 1.0), LIFELONG_CLIP)
        return weights

    def payload(self) -> dict[str, Any]:
        return {
            "schema": NOVELTY_SCHEMA,
            "alphabet": self.alphabet,
            "places_seen": len(self.lifelong),
            "distinct_readings": len(self.readings),
        }


class SlidingWindowUCB:
    """Agent57's meta-controller: pick a horizon, then judge it by results.

    A family of policies at different discounts is not a hyperparameter sweep
    done cheaply. Which horizon is right *changes during the run*: early on,
    everything nearby is unseen and a short horizon is enough; later the only
    novelty left is far away and a short horizon cannot see it. So the choice
    has to be made repeatedly against recent evidence, which is what the
    sliding window is for -- an ordinary bandit would average over a regime
    that has already ended.
    """

    def __init__(
        self,
        arms: Sequence[Any],
        *,
        window: int = 16,
        exploration: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not arms:
            raise ValueError("a bandit needs at least one arm")
        self.arms = tuple(arms)
        self.window = int(window)
        self.exploration = float(exploration)
        self.epsilon = float(epsilon)
        self._history: list[tuple[int, float]] = []
        self._generator = torch.Generator().manual_seed(int(seed))

    def _recent(self) -> list[tuple[int, float]]:
        return self._history[-self.window :]

    def select(self) -> int:
        recent = self._recent()
        counts = [0] * len(self.arms)
        totals = [0.0] * len(self.arms)
        for arm, reward in recent:
            counts[arm] += 1
            totals[arm] += reward
        untried = [arm for arm in range(len(self.arms)) if counts[arm] == 0]
        if untried:
            return untried[0]
        if float(torch.rand(1, generator=self._generator).item()) < self.epsilon:
            return int(
                torch.randint(0, len(self.arms), (1,), generator=self._generator).item()
            )
        total = len(recent)
        return max(
            range(len(self.arms)),
            key=lambda arm: totals[arm] / counts[arm]
            + self.exploration * math.sqrt(math.log(max(2, total)) / counts[arm]),
        )

    def update(self, arm: int, reward: float) -> None:
        if not 0 <= int(arm) < len(self.arms):
            raise ValueError("bandit arm is out of range")
        self._history.append((int(arm), float(reward)))

    def counts(self) -> tuple[int, ...]:
        tally = [0] * len(self.arms)
        for arm, _ in self._history:
            tally[arm] += 1
        return tuple(tally)

    def means(self) -> tuple[float, ...]:
        totals = [0.0] * len(self.arms)
        counts = [0] * len(self.arms)
        for arm, reward in self._history:
            totals[arm] += reward
            counts[arm] += 1
        return tuple(
            totals[arm] / counts[arm] if counts[arm] else 0.0
            for arm in range(len(self.arms))
        )


__all__ = [
    "LIFELONG_CLIP",
    "NOVELTY_SCHEMA",
    "NoveltyCounts",
    "SlidingWindowUCB",
]
