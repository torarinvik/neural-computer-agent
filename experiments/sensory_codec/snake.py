from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


DIRS = np.asarray([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)
OPPOSITE = (2, 3, 0, 1)


@dataclass
class SnakeState:
    snake: list[tuple[int, int]]
    apple: tuple[int, int]
    direction: int
    ate_last: bool = False
    done: bool = False


class SnakeEnv:
    """Small deterministic Snake environment with pixel-only observations.

    Coordinates are (row, column).  The observation deliberately flickers the
    apple and may hide the tail, making recent visual history useful.
    """

    def __init__(self, size: int = 10, seed: int = 0):
        if size < 7:
            raise ValueError("size must be at least 7")
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = self._new_state()

    def _new_state(self) -> SnakeState:
        row = int(self.rng.integers(2, self.size - 2))
        col = int(self.rng.integers(3, self.size - 2))
        snake = [(row, col), (row, col - 1), (row, col - 2)]
        state = SnakeState(snake=snake, apple=(0, 0), direction=1)
        self.state = state
        self._place_apple()
        return state

    def reset(self) -> SnakeState:
        self.steps = 0
        self.state = self._new_state()
        return self.state

    def _place_apple(self) -> None:
        occupied = set(self.state.snake)
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in occupied]
        self.state.apple = free[int(self.rng.integers(len(free)))]

    def danger(self, action: int) -> bool:
        if action == OPPOSITE[self.state.direction]:
            return True
        head = np.asarray(self.state.snake[0])
        nxt = tuple((head + DIRS[action]).tolist())
        if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size):
            return True
        # Moving into the current tail is safe when the tail will move away.
        body = self.state.snake[:-1] if nxt != self.state.apple else self.state.snake
        return nxt in body

    def step(self, action: int) -> tuple[float, bool]:
        if self.state.done:
            return 0.0, True
        if self.danger(action):
            self.state.done = True
            return -1.0, True
        self.state.direction = action
        nxt = tuple((np.asarray(self.state.snake[0]) + DIRS[action]).tolist())
        ate = nxt == self.state.apple
        self.state.snake.insert(0, nxt)
        if not ate:
            self.state.snake.pop()
        else:
            self._place_apple()
        self.state.ate_last = ate
        self.steps += 1
        return (1.0 if ate else 0.01), False

    def teacher_action(self) -> int:
        """BFS toward the apple, falling back to the safest open direction."""
        legal = [a for a in range(4) if not self.danger(a)]
        if not legal:
            return self.state.direction
        blocked = set(self.state.snake[:-1])
        start, goal = self.state.snake[0], self.state.apple
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal and path:
                return path[0]
            for action in legal if not path else range(4):
                nxt = tuple((np.asarray(pos) + DIRS[action]).tolist())
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in blocked and nxt not in seen):
                    seen.add(nxt)
                    queue.append((nxt, path + [action]))
        # Prefer moves with the largest flood-fill region.
        return max(legal, key=self._reachable_space)

    def _reachable_space(self, action: int) -> int:
        start = tuple((np.asarray(self.state.snake[0]) + DIRS[action]).tolist())
        blocked = set(self.state.snake[:-1])
        queue, seen = deque([start]), {start}
        while queue:
            pos = queue.popleft()
            for delta in DIRS:
                nxt = tuple((np.asarray(pos) + delta).tolist())
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in blocked and nxt not in seen):
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)

    def observe(self, apple_visible: bool = True, tail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        """Return 4xHxW pixels: walls, body, head, apple.

        Theme changes intensities without changing channel meaning. It is useful
        for testing superficial visual overfitting.
        """
        image = np.zeros((4, self.size + 2, self.size + 2), dtype=np.float32)
        image[0, 0, :] = image[0, -1, :] = 1.0
        image[0, :, 0] = image[0, :, -1] = 1.0
        body = self.state.snake if tail_visible else self.state.snake[:max(1, len(self.state.snake) // 2)]
        for row, col in body[1:]:
            image[1, row + 1, col + 1] = 1.0
        row, col = self.state.snake[0]
        image[2, row + 1, col + 1] = 1.0
        if apple_visible:
            row, col = self.state.apple
            image[3, row + 1, col + 1] = 1.0
        if theme:
            scales = np.asarray([1.0, 0.55, 0.75, 0.65], dtype=np.float32)
            image *= scales[:, None, None]
        return image

    def labels(self) -> dict[str, np.ndarray | int]:
        head = self.state.snake[0]
        apple = self.state.apple
        horizontal = 0 if apple[1] < head[1] else 2 if apple[1] > head[1] else 1
        vertical = 0 if apple[0] < head[0] else 2 if apple[0] > head[0] else 1
        return {
            "action": self.teacher_action(),
            "horizontal": horizontal,
            "vertical": vertical,
            "direction": self.state.direction,
            "danger": np.asarray([self.danger(a) for a in range(4)], dtype=np.float32),
            "ate": int(self.state.ate_last),
        }

    def semantic_vector(self) -> np.ndarray:
        """Privileged 16-D state used only to pretrain the grounded listener."""
        head = self.state.snake[0]
        apple = self.state.apple
        scale = max(1, self.size - 1)
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = (apple[1] - head[1]) / scale
        vec[1] = (apple[0] - head[0]) / scale
        vec[2:6] = [self.danger(a) for a in range(4)]
        vec[6 + self.state.direction] = 1.0
        vec[10] = len(self.state.snake) / (self.size * self.size)
        vec[11] = head[0] / scale
        vec[12] = head[1] / scale
        vec[13] = apple[0] / scale
        vec[14] = apple[1] / scale
        vec[15] = float(self.state.ate_last)
        return vec


def make_dataset(samples: int, sequence: int, size: int, seed: int,
                 partial: bool = True, themes: Iterable[int] = (0,)) -> dict[str, np.ndarray]:
    """Generate reproducible teacher rollouts; exact state never enters frame input."""
    rng = np.random.default_rng(seed)
    env = SnakeEnv(size=size, seed=seed)
    frames, semantic = [], []
    labels: dict[str, list] = {k: [] for k in ("action", "horizontal", "vertical", "direction", "danger", "ate")}
    themes = tuple(themes)
    while len(frames) < samples:
        env.reset()
        history: deque[np.ndarray] = deque(maxlen=sequence)
        for _ in range(300):
            # Apple is hidden in about one third of frames; the first half of a
            # long body is occasionally all that is visible.
            visible = not partial or rng.random() > 0.34
            tail_visible = not partial or rng.random() > 0.18
            theme = themes[int(rng.integers(len(themes)))]
            history.append(env.observe(visible, tail_visible, theme))
            if len(history) == sequence:
                frames.append(np.stack(history))
                semantic.append(env.semantic_vector())
                current = env.labels()
                for key in labels:
                    labels[key].append(current[key])
                if len(frames) >= samples:
                    break
            _, done = env.step(env.teacher_action())
            if done:
                break
    result = {"frames": np.stack(frames), "semantic": np.stack(semantic)}
    result.update({key: np.asarray(value) for key, value in labels.items()})
    return result

