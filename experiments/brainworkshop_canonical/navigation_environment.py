"""A world the agent's actions actually move it through.

Every environment in this repository generates its symbol stream in advance.
Whatever the agent answers, the next observation is the same -- so an action has
no consequence, there is nothing to plan, and "agent" has been a courtesy title
for a classifier reading a tape.

Here the tape is gone. The agent occupies one of the rendered grid places, an
action moves it, and what it sees next is where it went. Reward arrives only
when it is standing on a place it was never told about.

Three properties keep this honest and comparable with everything before it.

**The frontend is unchanged.** A place is drawn by the same `render_position`
the n-back tasks use, through the same frozen encoders, so the alphabet is
discovered exactly as before and nothing about the observation channel is new.
What is new is only that the sequence of observations is now the agent's own
doing.

**The dynamics are sampled, not written.** `transitions[action][place]` is
drawn per task. The agent cannot assume an action means "left" or "clockwise",
because across tasks it means neither; it has to find out. A hand-written maze
would let a solver that knew the layout pass for a learner.

**Reward is sparse and unexplained.** The verifier returns one scalar. It never
says where the goal is, and nothing but standing on it distinguishes it -- so
the goal has to be stumbled upon before it can be sought, which is the part of
the problem a pre-generated stream cannot pose.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .rendered_environment import _GRID_POSITIONS, render_position

NAVIGATION_SCHEMA = "neural-computer.navigation-task.v1"
PLACE_COUNT = len(_GRID_POSITIONS)
DEFAULT_ACTIONS = 4
DEFAULT_FRAME_SIZE = 36


@dataclass(frozen=True)
class NavigationTask:
    """Where the actions lead, where the reward is, and where it starts."""

    transitions: tuple[tuple[int, ...], ...]   # [action][place] -> place
    goal: int
    start: int
    place_count: int = PLACE_COUNT
    schema: str = NAVIGATION_SCHEMA

    def validate(self) -> NavigationTask:
        if self.schema != NAVIGATION_SCHEMA:
            raise ValueError("unsupported navigation task schema")
        if self.place_count < 2:
            raise ValueError("a navigation task needs at least two places")
        if not self.transitions:
            raise ValueError("a navigation task needs at least one action")
        for row in self.transitions:
            if len(row) != self.place_count:
                raise ValueError("every action must be defined at every place")
            if any(not 0 <= place < self.place_count for place in row):
                raise ValueError("an action leaves the place set")
        if not 0 <= self.goal < self.place_count:
            raise ValueError("the goal is outside the place set")
        if not 0 <= self.start < self.place_count:
            raise ValueError("the start is outside the place set")
        return self

    @property
    def action_count(self) -> int:
        return len(self.transitions)

    def distances(self) -> tuple[int, ...]:
        """Shortest number of moves from each place to the goal.

        Verifier-side only. Used to say how good a plan was, never to make one.
        """

        self.validate()
        unreachable = self.place_count + 1
        distance = [unreachable] * self.place_count
        distance[self.goal] = 0
        frontier = deque([self.goal])
        while frontier:
            place = frontier.popleft()
            for action in range(self.action_count):
                for source in range(self.place_count):
                    if (
                        int(self.transitions[action][source]) == place
                        and distance[source] == unreachable
                    ):
                        distance[source] = distance[place] + 1
                        frontier.append(source)
        return tuple(distance)

    def optimal_return(self, steps: int) -> float:
        """The best mean reward available in an episode of this length.

        Reward is paid on the step the agent *arrives*, so a goal `d` moves
        away pays on steps `d` through `steps` -- that is `steps - d + 1`
        paying steps, not `steps - d`. A first version dropped the arrival
        step and produced optimal returns that the agent then beat, which is
        how the off-by-one announced itself.
        """

        distance = self.distances()[self.start]
        if distance > steps:
            return 0.0
        return min(1.0, (steps - distance + 1) / steps)

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "transitions": [list(row) for row in self.transitions],
            "goal": int(self.goal),
            "start": int(self.start),
            "place_count": int(self.place_count),
        }


def sample_navigation_task(
    *,
    seed: int,
    place_count: int = PLACE_COUNT,
    action_count: int = DEFAULT_ACTIONS,
    minimum_distance: int = 2,
    attempts: int = 256,
) -> NavigationTask | None:
    """Draw a task that is worth solving and possible to solve.

    Three rejections, each of them about measurability rather than content:

    - the goal must be **reachable** from the start, or the task measures
      nothing;
    - it must be at least `minimum_distance` moves away, or a single lucky
      action solves it and planning is not what is being tested;
    - some action must **hold** at the goal, so an agent that gets there can
      stay and the best achievable return is well defined.
    """

    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(attempts):
        transitions = tuple(
            tuple(
                int(value)
                for value in torch.randint(
                    0, place_count, (place_count,), generator=generator
                )
            )
            for _ in range(action_count)
        )
        goal = int(torch.randint(0, place_count, (1,), generator=generator).item())
        start = int(torch.randint(0, place_count, (1,), generator=generator).item())
        if not any(int(row[goal]) == goal for row in transitions):
            continue
        candidate = NavigationTask(
            transitions=transitions,
            goal=goal,
            start=start,
            place_count=place_count,
        ).validate()
        distance = candidate.distances()[start]
        if distance > place_count or distance < minimum_distance:
            continue
        return candidate
    return None


@dataclass(frozen=True)
class NavigationStep:
    reward: torch.Tensor
    eligible: torch.Tensor


class NavigationVerifier:
    """Renders where the agent is; moves it; says only whether that paid.

    The place is verifier-private in the same sense the n-back target was: a
    caller can look at the picture and submit an action, and nothing in this
    interface will tell it which place it is at, where the goal is, or what an
    action does.
    """

    def __init__(
        self,
        task: NavigationTask,
        *,
        steps: int,
        frame_size: int = DEFAULT_FRAME_SIZE,
        frame_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        if steps < 1:
            raise ValueError("a navigation episode needs at least one step")
        if not 0.0 <= frame_noise < 1.0:
            raise ValueError("frame noise must be a non-negative fraction below one")
        self.task = task.validate()
        self.steps = int(steps)
        self.frame_size = int(frame_size)
        self.frame_noise = float(frame_noise)
        self._place = int(task.start)
        self._position = 0
        self._noise = torch.Generator().manual_seed(int(seed) + 977)

    @property
    def action_count(self) -> int:
        return self.task.action_count

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    def observation(self) -> torch.Tensor:
        """The rendered place, and nothing else."""

        if self.done:
            raise RuntimeError("navigation episode is complete")
        frame = render_position(self._place, size=self.frame_size)
        if self.frame_noise <= 0.0:
            return frame
        noise = torch.randn(frame.shape, generator=self._noise)
        return (frame + self.frame_noise * noise).clamp(0.0, 1.0)

    def score(self, action: torch.Tensor) -> NavigationStep:
        """Move, then say whether the agent landed on the goal."""

        if self.done:
            raise RuntimeError("navigation episode is complete")
        if action.shape != (1,) or action.dtype != torch.long:
            raise ValueError("navigation action must be int64 with shape [1]")
        chosen = int(action.item())
        if not 0 <= chosen < self.action_count:
            raise ValueError("navigation action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._position += 1
        return NavigationStep(
            reward=torch.tensor([float(self._place == self.task.goal)]),
            eligible=torch.tensor([True]),
        )
