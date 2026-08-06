"""Verifier-private game environments for the amodal game-playing rung.

Each verifier owns its full game state and exposes only a rendered float
observation grid plus a deterministic scalar outcome per attempted action.
The amodal controller never receives coordinates, entity labels, or the
reward rule; a caller-owned frontend must encode the raw observation into an
opaque learned event tensor before it reaches the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SNAKE_ACTION_COUNT = 4
PONG_ACTION_COUNT = 3

_SNAKE_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GameStep:
    """Verifier outcome for one attempted action."""

    reward: torch.Tensor
    alive: torch.Tensor


class SnakeVerifier:
    """Batched snake on a private grid with wall and self collisions.

    Actions are absolute headings: 0 up, 1 right, 2 down, 3 left.  A reversal
    into the snake's own neck is treated as continuing straight, matching the
    common playable rule.  Reward is +1 for eating food, -1 for dying, and 0
    otherwise; dead rows stay dead with zero reward.
    """

    action_count = SNAKE_ACTION_COUNT

    def __init__(
        self,
        *,
        batch_size: int,
        height: int = 8,
        width: int = 8,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if min(batch_size, height, width) < 4 and min(height, width) < 4:
            raise ValueError("grid must be at least 4x4")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._bodies: list[list[tuple[int, int]]] = []
        self._headings: list[int] = []
        self._food: list[tuple[int, int]] = []
        self._alive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    @property
    def observation_channels(self) -> int:
        return 3

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        center = (self.height // 2, self.width // 2)
        self._bodies = [
            [center, (center[0], center[1] - 1)] for _ in range(self.batch_size)
        ]
        self._headings = [1] * self.batch_size
        self._alive = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self._food = [self._sample_food(row) for row in range(self.batch_size)]

    def _sample_food(self, row: int) -> tuple[int, int]:
        occupied = set(self._bodies[row])
        while True:
            index = int(
                torch.randint(
                    0,
                    self.height * self.width,
                    (1,),
                    generator=self._generator,
                    device=self.device,
                ).item()
            )
            cell = (index // self.width, index % self.width)
            if cell not in occupied:
                return cell

    def observation(self) -> torch.Tensor:
        """Render [batch, 3, height, width] planes: body, head, food."""

        grid = torch.zeros(
            self.batch_size,
            3,
            self.height,
            self.width,
            device=self.device,
        )
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            for cell in self._bodies[row]:
                grid[row, 0, cell[0], cell[1]] = 1.0
            head = self._bodies[row][0]
            grid[row, 1, head[0], head[1]] = 1.0
            food = self._food[row]
            grid[row, 2, food[0], food[1]] = 1.0
        return grid

    def step(self, actions: torch.Tensor) -> GameStep:
        if actions.shape != (self.batch_size,):
            raise ValueError("actions must have shape [batch]")
        if bool((actions < 0).any()) or bool((actions >= self.action_count).any()):
            raise ValueError("action out of range")
        reward = torch.zeros(self.batch_size, device=self.device)
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            heading = int(actions[row].item())
            if (heading + 2) % 4 == self._headings[row]:
                heading = self._headings[row]
            self._headings[row] = heading
            delta = _SNAKE_DELTAS[heading]
            head = self._bodies[row][0]
            target = (head[0] + delta[0], head[1] + delta[1])
            grows = target == self._food[row]
            body = self._bodies[row] if grows else self._bodies[row][:-1]
            hits_wall = not (
                0 <= target[0] < self.height and 0 <= target[1] < self.width
            )
            if hits_wall or target in body:
                self._alive[row] = False
                reward[row] = -1.0
                continue
            self._bodies[row] = [target, *body]
            if grows:
                reward[row] = 1.0
                self._food[row] = self._sample_food(row)
        return GameStep(reward=reward, alive=self._alive.clone())


class PongVerifier:
    """Batched single-paddle pong on a private grid.

    The ball moves one cell per step and reflects off the side and top walls.
    The paddle occupies ``paddle_width`` cells on the bottom row; actions are
    0 left, 1 stay, 2 right.  Reward is +1 when the paddle returns the ball
    at the bottom row and -1 when it misses, which ends that row's episode.
    """

    action_count = PONG_ACTION_COUNT

    def __init__(
        self,
        *,
        batch_size: int,
        height: int = 8,
        width: int = 8,
        paddle_width: int = 2,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if min(height, width) < 4:
            raise ValueError("grid must be at least 4x4")
        if not 1 <= paddle_width < width:
            raise ValueError("paddle width must fit inside the grid")
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.paddle_width = int(paddle_width)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._ball = torch.zeros(self.batch_size, 2, dtype=torch.long, device=self.device)
        self._velocity = torch.zeros(
            self.batch_size, 2, dtype=torch.long, device=self.device
        )
        self._paddle = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self._alive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    @property
    def observation_channels(self) -> int:
        return 2

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        self._ball[:, 0] = 0
        self._ball[:, 1] = torch.randint(
            0,
            self.width,
            (self.batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._velocity[:, 0] = 1
        self._velocity[:, 1] = (
            torch.randint(
                0,
                2,
                (self.batch_size,),
                generator=self._generator,
                device=self.device,
            )
            * 2
            - 1
        )
        self._paddle = torch.randint(
            0,
            self.width - self.paddle_width + 1,
            (self.batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._alive = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def observation(self) -> torch.Tensor:
        """Render [batch, 2, height, width] planes: ball, paddle."""

        grid = torch.zeros(
            self.batch_size,
            2,
            self.height,
            self.width,
            device=self.device,
        )
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            grid[row, 0, int(self._ball[row, 0]), int(self._ball[row, 1])] = 1.0
            start = int(self._paddle[row])
            grid[row, 1, self.height - 1, start : start + self.paddle_width] = 1.0
        return grid

    def step(self, actions: torch.Tensor) -> GameStep:
        if actions.shape != (self.batch_size,):
            raise ValueError("actions must have shape [batch]")
        if bool((actions < 0).any()) or bool((actions >= self.action_count).any()):
            raise ValueError("action out of range")
        reward = torch.zeros(self.batch_size, device=self.device)
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            move = int(actions[row].item()) - 1
            self._paddle[row] = int(
                min(
                    max(int(self._paddle[row]) + move, 0),
                    self.width - self.paddle_width,
                )
            )
            ball_y = int(self._ball[row, 0]) + int(self._velocity[row, 0])
            ball_x = int(self._ball[row, 1]) + int(self._velocity[row, 1])
            if ball_x < 0 or ball_x >= self.width:
                self._velocity[row, 1] = -self._velocity[row, 1]
                ball_x = min(max(ball_x, 0), self.width - 1)
            if ball_y < 0:
                self._velocity[row, 0] = 1
                ball_y = 1
            if ball_y >= self.height - 1:
                start = int(self._paddle[row])
                if start <= ball_x < start + self.paddle_width:
                    self._velocity[row, 0] = -1
                    ball_y = self.height - 2
                    reward[row] = 1.0
                else:
                    self._alive[row] = False
                    reward[row] = -1.0
                    continue
            self._ball[row, 0] = ball_y
            self._ball[row, 1] = ball_x
        return GameStep(reward=reward, alive=self._alive.clone())


BREAKOUT_ACTION_COUNT = 3


class BreakoutVerifier:
    """Batched single-paddle breakout on a private grid.

    Bricks fill the top rows; the ball reflects off walls, the paddle, and
    bricks.  Actions are 0 left, 1 stay, 2 right.  Reward is +1 for each
    brick broken and -1 for a miss at the bottom row, which ends that row's
    episode.  Clearing every brick also ends the episode (with the final
    brick's reward), so lifetimes are bounded without a step cap.
    """

    action_count = BREAKOUT_ACTION_COUNT

    def __init__(
        self,
        *,
        batch_size: int,
        height: int = 8,
        width: int = 8,
        paddle_width: int = 2,
        brick_rows: int = 2,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if min(height, width) < 6:
            raise ValueError("grid must be at least 6x6")
        if not 1 <= paddle_width < width:
            raise ValueError("paddle width must fit inside the grid")
        if not 1 <= brick_rows <= height - 4:
            raise ValueError("brick rows must leave room for play")
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.paddle_width = int(paddle_width)
        self.brick_rows = int(brick_rows)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._ball = torch.zeros(self.batch_size, 2, dtype=torch.long, device=self.device)
        self._velocity = torch.zeros(
            self.batch_size, 2, dtype=torch.long, device=self.device
        )
        self._paddle = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self._bricks = torch.zeros(
            self.batch_size,
            self.brick_rows,
            self.width,
            dtype=torch.bool,
            device=self.device,
        )
        self._alive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    @property
    def observation_channels(self) -> int:
        return 3

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        self._bricks.fill_(True)
        self._ball[:, 0] = self.brick_rows + 1
        self._ball[:, 1] = torch.randint(
            0,
            self.width,
            (self.batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._velocity[:, 0] = 1
        self._velocity[:, 1] = (
            torch.randint(
                0,
                2,
                (self.batch_size,),
                generator=self._generator,
                device=self.device,
            )
            * 2
            - 1
        )
        self._paddle = torch.randint(
            0,
            self.width - self.paddle_width + 1,
            (self.batch_size,),
            generator=self._generator,
            device=self.device,
        )
        self._alive = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def observation(self) -> torch.Tensor:
        """Render [batch, 3, height, width] planes: ball, paddle, bricks."""

        grid = torch.zeros(
            self.batch_size,
            3,
            self.height,
            self.width,
            device=self.device,
        )
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            grid[row, 0, int(self._ball[row, 0]), int(self._ball[row, 1])] = 1.0
            start = int(self._paddle[row])
            grid[row, 1, self.height - 1, start : start + self.paddle_width] = 1.0
            grid[row, 2, : self.brick_rows] = self._bricks[row].float()
        return grid

    def step(self, actions: torch.Tensor) -> GameStep:
        if actions.shape != (self.batch_size,):
            raise ValueError("actions must have shape [batch]")
        if bool((actions < 0).any()) or bool((actions >= self.action_count).any()):
            raise ValueError("action out of range")
        reward = torch.zeros(self.batch_size, device=self.device)
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            move = int(actions[row].item()) - 1
            self._paddle[row] = int(
                min(
                    max(int(self._paddle[row]) + move, 0),
                    self.width - self.paddle_width,
                )
            )
            ball_y = int(self._ball[row, 0]) + int(self._velocity[row, 0])
            ball_x = int(self._ball[row, 1]) + int(self._velocity[row, 1])
            if ball_x < 0 or ball_x >= self.width:
                self._velocity[row, 1] = -self._velocity[row, 1]
                ball_x = min(max(ball_x, 0), self.width - 1)
            if ball_y < 0:
                self._velocity[row, 0] = 1
                ball_y = min(1, self.height - 1)
            if ball_y < self.brick_rows and bool(self._bricks[row, ball_y, ball_x]):
                self._bricks[row, ball_y, ball_x] = False
                self._velocity[row, 0] = -self._velocity[row, 0]
                reward[row] = 1.0
                if not bool(self._bricks[row].any()):
                    self._alive[row] = False
                continue
            if ball_y >= self.height - 1:
                start = int(self._paddle[row])
                if start <= ball_x < start + self.paddle_width:
                    self._velocity[row, 0] = -1
                    ball_y = self.height - 2
                else:
                    self._alive[row] = False
                    reward[row] = -1.0
                    continue
            self._ball[row, 0] = ball_y
            self._ball[row, 1] = ball_x
        return GameStep(reward=reward, alive=self._alive.clone())
