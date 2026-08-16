"""Procedurally generated rendered mazes for the shared agent.

The maze is a downstream task, not a second controller.  The verifier owns
the wall layout, the protocol-to-direction permutation, the goal, and the
current cell.  Public observations contain only a pixel rerender of the
walls and the agent.  Reward is a scalar paid after arrival at the hidden
goal, so an agent must explore, model the action-conditioned world, and then
plan through that model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

MAZE_SCHEMA = "neural-computer.maze-task.v1"
DEFAULT_GRID_SIZE = 7
DEFAULT_ACTION_COUNT = 4
DEFAULT_FRAME_SIZE = 42

# Protocol actions are deliberately opaque.  A task may permute these
# directions, so an action index never means ``north`` across tasks.
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _perfect_maze(grid_size: int, generator: torch.Generator) -> tuple[tuple[int, ...], ...]:
    """Carve one connected odd-grid maze and return its open-cell mask."""

    walls = [[True] * grid_size for _ in range(grid_size)]
    start = (1, 1)
    walls[start[0]][start[1]] = False
    frontier = [start]
    while frontier:
        index = int(torch.randint(len(frontier), (1,), generator=generator).item())
        row, column = frontier.pop(index)
        candidates = []
        for delta_row, delta_column in _DIRECTIONS:
            next_row = row + 2 * delta_row
            next_column = column + 2 * delta_column
            if (
                0 < next_row < grid_size - 1
                and 0 < next_column < grid_size - 1
                and walls[next_row][next_column]
            ):
                candidates.append((next_row, next_column, delta_row, delta_column))
        # Randomised depth-first carving.  A frontier cell is only opened once,
        # which keeps the topology a tree while making every open cell reachable.
        for next_row, next_column, delta_row, delta_column in candidates:
            if not walls[next_row][next_column]:
                continue
            walls[row + delta_row][column + delta_column] = False
            walls[next_row][next_column] = False
            frontier.append((next_row, next_column))
    return tuple(tuple(int(value) for value in row) for row in walls)


@dataclass(frozen=True)
class MazeTask:
    """A hidden-goal maze with an opaque action permutation."""

    walls: tuple[tuple[int, ...], ...]
    open_positions: tuple[tuple[int, int], ...]
    transitions: tuple[tuple[int, ...], ...]
    action_permutation: tuple[int, ...]
    goal: int
    start: int
    grid_size: int = DEFAULT_GRID_SIZE
    schema: str = MAZE_SCHEMA

    def validate(self) -> MazeTask:
        if self.schema != MAZE_SCHEMA:
            raise ValueError("unsupported maze task schema")
        if self.grid_size < 5 or self.grid_size % 2 == 0:
            raise ValueError("maze grid size must be odd and at least five")
        if len(self.walls) != self.grid_size or any(
            len(row) != self.grid_size for row in self.walls
        ):
            raise ValueError("maze wall mask has the wrong shape")
        if len(self.open_positions) < 2:
            raise ValueError("maze needs at least two open cells")
        if len(self.transitions) != DEFAULT_ACTION_COUNT:
            raise ValueError("maze protocol must expose four opaque actions")
        if len(self.action_permutation) != DEFAULT_ACTION_COUNT or sorted(
            self.action_permutation
        ) != list(range(DEFAULT_ACTION_COUNT)):
            raise ValueError("maze action permutation must be a direction permutation")
        place_count = len(self.open_positions)
        if any(len(row) != place_count for row in self.transitions):
            raise ValueError("maze transition rows have the wrong length")
        if any(
            following < 0 or following >= place_count
            for row in self.transitions
            for following in row
        ):
            raise ValueError("maze transition leaves the open-cell set")
        if not 0 <= self.goal < place_count or not 0 <= self.start < place_count:
            raise ValueError("maze start or goal is outside the open-cell set")
        if self.start == self.goal:
            raise ValueError("maze start and goal must differ")
        return self

    @property
    def place_count(self) -> int:
        return len(self.open_positions)

    @property
    def action_count(self) -> int:
        return DEFAULT_ACTION_COUNT

    def distances(self) -> tuple[int, ...]:
        """Verifier-side shortest distances; never supplied to the agent."""

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
                        self.transitions[action][source] == place
                        and distance[source] == unreachable
                    ):
                        distance[source] = distance[place] + 1
                        frontier.append(source)
        return tuple(distance)

    def optimal_return(self, steps: int) -> float:
        distance = self.distances()[self.start]
        if distance > steps:
            return 0.0
        return min(1.0, (steps - distance + 1) / steps)

    def with_start(self, start: int) -> MazeTask:
        candidate = MazeTask(
            walls=self.walls,
            open_positions=self.open_positions,
            transitions=self.transitions,
            action_permutation=self.action_permutation,
            goal=self.goal,
            start=int(start),
            grid_size=self.grid_size,
        )
        return candidate.validate()

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "grid_size": self.grid_size,
            "walls": [list(row) for row in self.walls],
            "open_positions": [list(position) for position in self.open_positions],
            "transitions": [list(row) for row in self.transitions],
            "action_permutation": list(self.action_permutation),
            "goal": self.goal,
            "start": self.start,
        }


def sample_maze_task(
    *,
    seed: int,
    grid_size: int = DEFAULT_GRID_SIZE,
    minimum_distance: int = 6,
    attempts: int = 256,
) -> MazeTask | None:
    """Sample a connected maze whose hidden goal requires planning."""

    if grid_size < 5 or grid_size % 2 == 0:
        raise ValueError("maze grid size must be odd and at least five")
    if minimum_distance < 1:
        raise ValueError("maze minimum distance must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(int(attempts)):
        walls = _perfect_maze(grid_size, generator)
        positions = tuple(
            (row, column)
            for row in range(grid_size)
            for column in range(grid_size)
            if not walls[row][column]
        )
        position_to_place = {position: index for index, position in enumerate(positions)}
        permutation = tuple(
            int(value)
            for value in torch.randperm(DEFAULT_ACTION_COUNT, generator=generator)
        )
        transitions = []
        for action in range(DEFAULT_ACTION_COUNT):
            direction = _DIRECTIONS[permutation[action]]
            row = []
            for current_row, current_column in positions:
                following = (
                    current_row + direction[0],
                    current_column + direction[1],
                )
                row.append(position_to_place.get(following, position_to_place[(current_row, current_column)]))
            transitions.append(tuple(row))
        goal = int(torch.randint(len(positions), (1,), generator=generator).item())
        start = int(torch.randint(len(positions), (1,), generator=generator).item())
        task = MazeTask(
            walls=walls,
            open_positions=positions,
            transitions=tuple(transitions),
            action_permutation=permutation,
            goal=goal,
            start=start,
            grid_size=grid_size,
        ).validate()
        if task.distances()[start] >= minimum_distance:
            return task
    return None


def render_maze(
    task: MazeTask,
    place: int,
    *,
    size: int = DEFAULT_FRAME_SIZE,
) -> torch.Tensor:
    """Render walls and the current cell, without the hidden goal."""

    task.validate()
    if size < task.grid_size or size % task.grid_size:
        raise ValueError("maze frame size must be divisible by the grid size")
    if not 0 <= place < task.place_count:
        raise ValueError("maze place is outside the open-cell set")
    frame = torch.zeros(3, size, size)
    cell = size // task.grid_size
    for row in range(task.grid_size):
        for column in range(task.grid_size):
            top = row * cell
            left = column * cell
            color = 0.06 if task.walls[row][column] else 0.78
            frame[:, top : top + cell, left : left + cell] = color
    row, column = task.open_positions[place]
    margin = max(1, cell // 5)
    top = row * cell + margin
    left = column * cell + margin
    bottom = (row + 1) * cell - margin
    right = (column + 1) * cell - margin
    frame[:, top:bottom, left:right] = torch.tensor((0.25, 0.70, 1.0)).view(3, 1, 1)
    return frame


@dataclass(frozen=True)
class MazeStep:
    reward: torch.Tensor
    eligible: torch.Tensor


class MazeVerifier:
    """Private maze state with a public rendered frame and scalar outcome."""

    def __init__(
        self,
        task: MazeTask,
        *,
        steps: int,
        frame_size: int = DEFAULT_FRAME_SIZE,
    ) -> None:
        if steps < 1:
            raise ValueError("maze episode needs at least one step")
        self.task = task.validate()
        self.steps = int(steps)
        self.frame_size = int(frame_size)
        self._place = int(task.start)
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    def observation(self) -> torch.Tensor:
        # A terminal frame is still a public pixel observation.  Exposing it
        # lets the learner attribute the final action-conditioned transition
        # and its scalar reward without leaking the hidden goal or coordinates.
        return render_maze(self.task, self._place, size=self.frame_size)

    def score(self, action: torch.Tensor) -> MazeStep:
        if self.done:
            raise RuntimeError("maze episode is complete")
        if action.shape != (1,) or action.dtype != torch.long:
            raise ValueError("maze action must be int64 with shape [1]")
        chosen = int(action.item())
        if not 0 <= chosen < self.task.action_count:
            raise ValueError("maze action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._position += 1
        return MazeStep(
            reward=torch.tensor(
                [float(self._place == self.task.goal)], dtype=torch.float32
            ),
            eligible=torch.tensor([True]),
        )
