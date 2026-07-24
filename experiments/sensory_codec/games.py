from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .snake import DIRS, OPPOSITE, SnakeEnv


GAME_NAMES = (
    "snake", "collect", "dodge", "chase",
    "maze", "keydoor", "memory", "patrol", "signal", "rhythm",
)

# Families deliberately overlap. Holding one out removes every game that teaches
# that capability, which is stricter than leaving out a single named environment.
CAPABILITY_FAMILIES = {
    "audio_command": ("signal", "rhythm"),
    "text_state": ("keydoor", "signal", "rhythm"),
    "persistent_memory": ("snake", "keydoor", "memory"),
    "velocity_prediction": ("dodge", "chase", "patrol"),
    "route_planning": ("collect", "maze", "keydoor"),
}


class PrimitiveGame(Protocol):
    size: int
    direction: int
    event_last: bool
    done: bool

    def reset(self): ...
    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray: ...
    def teacher_action(self) -> int: ...
    def step(self, action: int) -> tuple[float, bool]: ...
    def labels(self) -> dict[str, np.ndarray | int]: ...
    def semantic_vector(self) -> np.ndarray: ...
    def raw_audio(self, samples: int = 64) -> np.ndarray: ...
    def raw_text(self, characters: int = 32) -> np.ndarray: ...


def _relative_labels(agent: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    horizontal = 0 if target[1] < agent[1] else 2 if target[1] > agent[1] else 1
    vertical = 0 if target[0] < agent[0] else 2 if target[0] > agent[0] else 1
    return horizontal, vertical


def _semantic(size: int, agent: tuple[int, int], target: tuple[int, int],
              danger: list[bool], direction: int, progress: float,
              event_last: bool) -> np.ndarray:
    scale = max(1, size - 1)
    vec = np.zeros(16, dtype=np.float32)
    vec[0] = (target[1] - agent[1]) / scale
    vec[1] = (target[0] - agent[0]) / scale
    vec[2:6] = danger
    vec[6 + direction] = 1.0
    vec[10] = progress
    vec[11] = agent[0] / scale
    vec[12] = agent[1] / scale
    vec[13] = target[0] / scale
    vec[14] = target[1] / scale
    vec[15] = float(event_last)
    return vec


class BaseGridGame:
    def __init__(self, size: int = 10, seed: int = 0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.direction = 1
        self.event_last = False
        self.done = False
        self.steps = 0

    def _base_image(self) -> np.ndarray:
        image = np.zeros((4, self.size + 2, self.size + 2), dtype=np.float32)
        image[0, 0, :] = image[0, -1, :] = 1.0
        image[0, :, 0] = image[0, :, -1] = 1.0
        return image

    @staticmethod
    def _theme(image: np.ndarray, theme: int) -> np.ndarray:
        if theme:
            image *= np.asarray([1.0, 0.55, 0.75, 0.65], dtype=np.float32)[:, None, None]
        return image

    def labels(self) -> dict[str, np.ndarray | int]:
        horizontal, vertical = _relative_labels(self.agent, self.target)
        return {
            "action": self.teacher_action(),
            "horizontal": horizontal,
            "vertical": vertical,
            "direction": self.direction,
            "danger": np.asarray([self.danger(a) for a in range(4)], dtype=np.float32),
            "ate": int(self.event_last),
        }

    def semantic_vector(self) -> np.ndarray:
        return _semantic(self.size, self.agent, self.target,
                         [self.danger(a) for a in range(4)], self.direction,
                         self.progress(), self.event_last)

    def progress(self) -> float:
        return min(1.0, self.steps / 250.0)

    def raw_audio(self, samples: int = 64) -> np.ndarray:
        """Rendered PCM-like sensor. Default is silence plus an event chirp."""
        if not self.event_last:
            return np.zeros(samples, dtype=np.float32)
        phase = np.linspace(0, 8 * np.pi, samples, endpoint=False)
        return (0.5 * np.sin(phase)).astype(np.float32)

    def raw_text(self, characters: int = 32) -> np.ndarray:
        """Raw visible character cells; zero is blank, never privileged narration."""
        return np.zeros(characters, dtype=np.int64)


class CollectEnv(BaseGridGame):
    """Static obstacle planning and repeated target collection."""

    def __init__(self, size: int = 10, seed: int = 0):
        super().__init__(size, seed)
        self.reset()

    def reset(self):
        self.direction, self.event_last, self.done, self.steps = 1, False, False, 0
        self.agent = (self.size // 2, self.size // 2)
        candidates = [(r, c) for r in range(self.size) for c in range(self.size)
                      if (r, c) != self.agent]
        self.walls = set()
        self.target = candidates[int(self.rng.integers(len(candidates)))]
        for _ in range(max(2, self.size // 2)):
            wall = candidates[int(self.rng.integers(len(candidates)))]
            if wall != self.target:
                self.walls.add(wall)
        self.visited = deque([self.agent], maxlen=8)
        return self

    def danger(self, action: int) -> bool:
        nxt = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        return not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size) or nxt in self.walls

    def _place_target(self) -> None:
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) != self.agent and (r, c) not in self.walls]
        self.target = free[int(self.rng.integers(len(free)))]

    def teacher_action(self) -> int:
        legal = [a for a in range(4) if not self.danger(a)]
        queue = deque([(self.agent, [])])
        seen = {self.agent}
        while queue:
            pos, path = queue.popleft()
            if pos == self.target and path:
                return path[0]
            for action in range(4):
                nxt = tuple((np.asarray(pos) + DIRS[action]).tolist())
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in self.walls and nxt not in seen):
                    seen.add(nxt)
                    queue.append((nxt, path + [action]))
        return legal[0] if legal else self.direction

    def step(self, action: int) -> tuple[float, bool]:
        if self.danger(action):
            self.done = True
            return -1.0, True
        self.direction = action
        self.agent = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        self.visited.append(self.agent)
        self.event_last = self.agent == self.target
        if self.event_last:
            self._place_target()
        self.steps += 1
        return (1.0 if self.event_last else 0.01), False

    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        image = self._base_image()
        for row, col in self.walls:
            image[0, row + 1, col + 1] = 1.0
        if detail_visible:
            for row, col in self.visited:
                image[1, row + 1, col + 1] = 0.45
        image[2, self.agent[0] + 1, self.agent[1] + 1] = 1.0
        if target_visible:
            image[3, self.target[0] + 1, self.target[1] + 1] = 1.0
        return self._theme(image, theme)


class DodgeEnv(BaseGridGame):
    """Navigate to an exit while hazards sweep horizontally."""

    def __init__(self, size: int = 10, seed: int = 0):
        super().__init__(size, seed)
        self.reset()

    def reset(self):
        self.direction, self.event_last, self.done, self.steps = 0, False, False, 0
        self.agent = (self.size - 1, self.size // 2)
        self.target = (0, self.size // 2)
        rows = list(range(2, self.size - 1, 2))
        self.hazards = [(row, int(self.rng.integers(self.size)), 1 if i % 2 == 0 else -1)
                        for i, row in enumerate(rows)]
        return self

    def _next_hazards(self) -> list[tuple[int, int, int]]:
        result = []
        for row, col, velocity in self.hazards:
            nxt = col + velocity
            if nxt < 0 or nxt >= self.size:
                velocity *= -1
                nxt = col + velocity
            result.append((row, nxt, velocity))
        return result

    def danger(self, action: int) -> bool:
        nxt = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size):
            return True
        return nxt in {(row, col) for row, col, _ in self._next_hazards()}

    def teacher_action(self) -> int:
        legal = [a for a in range(4) if not self.danger(a)]
        if not legal:
            return self.direction
        # Prefer upward progress, then proximity to the exit, while avoiding the
        # predicted next hazard positions supplied by the simulator.
        return min(legal, key=lambda a: (
            tuple((np.asarray(self.agent) + DIRS[a]).tolist())[0],
            abs(tuple((np.asarray(self.agent) + DIRS[a]).tolist())[1] - self.target[1]),
        ))

    def step(self, action: int) -> tuple[float, bool]:
        if self.danger(action):
            self.done = True
            return -1.0, True
        self.direction = action
        self.agent = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        self.hazards = self._next_hazards()
        self.event_last = self.agent == self.target
        self.steps += 1
        if self.event_last:
            self.agent = (self.size - 1, int(self.rng.integers(self.size)))
        return (1.0 if self.event_last else 0.01), False

    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        image = self._base_image()
        if detail_visible:
            for row, col, _ in self.hazards:
                image[1, row + 1, col + 1] = 1.0
        image[2, self.agent[0] + 1, self.agent[1] + 1] = 1.0
        if target_visible:
            image[3, self.target[0] + 1, self.target[1] + 1] = 1.0
        return self._theme(image, theme)


class ChaseEnv(BaseGridGame):
    """Intercept a bouncing target; its velocity is visible only through history."""

    def __init__(self, size: int = 10, seed: int = 0):
        super().__init__(size, seed)
        self.reset()

    def reset(self):
        self.direction, self.event_last, self.done, self.steps = 1, False, False, 0
        self.agent = (self.size // 2, self.size // 2)
        self.target = (1, 1)
        self.velocity = (1, 1)
        self.trail = deque([self.target], maxlen=4)
        return self

    def _next_target(self) -> tuple[tuple[int, int], tuple[int, int]]:
        row, col = self.target
        vr, vc = self.velocity
        if not 0 <= row + vr < self.size:
            vr *= -1
        if not 0 <= col + vc < self.size:
            vc *= -1
        return (row + vr, col + vc), (vr, vc)

    def danger(self, action: int) -> bool:
        nxt = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        return not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size)

    def teacher_action(self) -> int:
        predicted, _ = self._next_target()
        legal = [a for a in range(4) if not self.danger(a)]
        return min(legal, key=lambda a: sum(abs(x - y) for x, y in zip(
            tuple((np.asarray(self.agent) + DIRS[a]).tolist()), predicted)))

    def step(self, action: int) -> tuple[float, bool]:
        if self.danger(action):
            self.done = True
            return -1.0, True
        self.direction = action
        self.agent = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        self.target, self.velocity = self._next_target()
        self.trail.append(self.target)
        self.event_last = self.agent == self.target
        if self.event_last:
            self.target = (int(self.rng.integers(self.size)), int(self.rng.integers(self.size)))
            self.velocity = (1 if self.rng.random() > 0.5 else -1,
                             1 if self.rng.random() > 0.5 else -1)
            self.trail.clear()
            self.trail.append(self.target)
        self.steps += 1
        return (1.0 if self.event_last else 0.01), False

    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        image = self._base_image()
        if detail_visible:
            for row, col in self.trail:
                image[1, row + 1, col + 1] = 0.4
        image[2, self.agent[0] + 1, self.agent[1] + 1] = 1.0
        if target_visible:
            image[3, self.target[0] + 1, self.target[1] + 1] = 1.0
        return self._theme(image, theme)


class MissionEnv(BaseGridGame):
    """Six compositional games sharing sensors but stressing distinct capabilities."""

    MODES = ("maze", "keydoor", "memory", "patrol", "signal", "rhythm")

    def __init__(self, mode: str, size: int = 10, seed: int = 0):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        super().__init__(size, seed)
        self.reset()

    def reset(self):
        self.direction, self.event_last, self.done, self.steps = 0, False, False, 0
        self.agent = (self.size - 1, self.size // 2)
        self.target = (0, self.size // 2)
        self.walls: set[tuple[int, int]] = set()
        self.key = (self.size - 2, 1)
        self.has_key = False
        self.reveal_until = 5
        self.patrol = (self.size // 2, self.size // 2)
        self.signal_direction = int(self.rng.integers(4))
        self.beat_period = 4
        if self.mode == "maze":
            # Vertical walls with alternating gaps guarantee a route while making
            # the layout procedurally different across seeds.
            for col in range(2, self.size - 1, 2):
                gap = int(self.rng.integers(self.size))
                self.walls.update((row, col) for row in range(self.size) if row != gap)
        elif self.mode in ("keydoor",):
            self.walls = {(self.size // 2, col) for col in range(2, self.size - 1)}
            self.walls.discard((self.size // 2, self.size // 2))
        elif self.mode == "memory":
            self.target = (int(self.rng.integers(self.size // 2)),
                           int(self.rng.integers(self.size)))
        elif self.mode == "signal":
            self.agent = (self.size // 2, self.size // 2)
            self._set_signal_target()
        return self

    def _set_signal_target(self) -> None:
        row, col = self.agent
        delta = DIRS[self.signal_direction]
        distance = max(1, min(3, self.size - 1))
        self.target = (
            int(np.clip(row + delta[0] * distance, 0, self.size - 1)),
            int(np.clip(col + delta[1] * distance, 0, self.size - 1)),
        )

    def _subgoal(self) -> tuple[int, int]:
        if self.mode == "keydoor" and not self.has_key:
            return self.key
        return self.target

    def labels(self) -> dict[str, np.ndarray | int]:
        saved = self.target
        self.target = self._subgoal()
        result = super().labels()
        self.target = saved
        return result

    def semantic_vector(self) -> np.ndarray:
        saved = self.target
        self.target = self._subgoal()
        result = super().semantic_vector()
        self.target = saved
        return result

    def _next_patrol(self) -> tuple[int, int]:
        row, col = self.patrol
        ar, ac = self.agent
        candidates = []
        if row != ar:
            candidates.append((row + (1 if ar > row else -1), col))
        if col != ac:
            candidates.append((row, col + (1 if ac > col else -1)))
        return candidates[self.steps % len(candidates)] if candidates else self.patrol

    def danger(self, action: int) -> bool:
        nxt = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size) or nxt in self.walls:
            return True
        if self.mode == "patrol" and nxt == self._next_patrol():
            return True
        if self.mode == "rhythm":
            crossing_gate = self.agent[0] == self.size // 2 + 1 and nxt[0] == self.size // 2
            if crossing_gate and self.steps % self.beat_period != 0:
                return True
        return False

    def teacher_action(self) -> int:
        goal = self._subgoal()
        legal = [action for action in range(4) if not self.danger(action)]
        if not legal:
            return self.direction
        return min(legal, key=lambda action: sum(abs(a - b) for a, b in zip(
            tuple((np.asarray(self.agent) + DIRS[action]).tolist()), goal)))

    def step(self, action: int) -> tuple[float, bool]:
        if self.danger(action):
            self.done = True
            return -1.0, True
        next_patrol = self._next_patrol() if self.mode == "patrol" else self.patrol
        self.direction = action
        self.agent = tuple((np.asarray(self.agent) + DIRS[action]).tolist())
        if self.mode == "patrol":
            self.patrol = next_patrol
            if self.agent == self.patrol:
                self.done = True
                return -1.0, True
        if self.mode == "keydoor" and not self.has_key and self.agent == self.key:
            self.has_key = True
        self.event_last = self.agent == self.target and (self.mode != "keydoor" or self.has_key)
        self.steps += 1
        if self.event_last:
            if self.mode == "signal":
                self.signal_direction = int(self.rng.integers(4))
                self._set_signal_target()
            elif self.mode == "memory":
                self.target = (int(self.rng.integers(self.size)), int(self.rng.integers(self.size)))
                self.reveal_until = self.steps + 5
            else:
                self.agent = (self.size - 1, int(self.rng.integers(self.size)))
                self.has_key = False
        return (1.0 if self.event_last else 0.01), False

    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        image = self._base_image()
        for row, col in self.walls:
            image[0, row + 1, col + 1] = 1.0
        if detail_visible:
            if self.mode == "patrol":
                image[1, self.patrol[0] + 1, self.patrol[1] + 1] = 1.0
            elif self.mode == "keydoor" and not self.has_key:
                image[1, self.key[0] + 1, self.key[1] + 1] = 1.0
            elif self.mode == "rhythm":
                image[1, self.size // 2 + 1, 1:-1] = 0.35 + 0.15 * (self.steps % self.beat_period)
        image[2, self.agent[0] + 1, self.agent[1] + 1] = 1.0
        visible = target_visible
        if self.mode == "memory":
            visible = visible and self.steps < self.reveal_until
        if self.mode == "signal":
            visible = False
        if visible:
            goal = self._subgoal()
            image[3, goal[0] + 1, goal[1] + 1] = 1.0
        return self._theme(image, theme)

    def raw_audio(self, samples: int = 64) -> np.ndarray:
        phase = np.arange(samples, dtype=np.float32) / samples
        if self.mode == "signal":
            frequency = (self.signal_direction + 1) * 3
            return (0.35 * np.sin(2 * np.pi * frequency * phase)).astype(np.float32)
        if self.mode == "rhythm":
            amplitude = 0.8 if self.steps % self.beat_period == 0 else 0.08
            return (amplitude * np.sin(2 * np.pi * 6 * phase)).astype(np.float32)
        return super().raw_audio(samples)

    def raw_text(self, characters: int = 32) -> np.ndarray:
        if self.mode == "signal":
            message = ("GO UP", "GO RIGHT", "GO DOWN", "GO LEFT")[self.signal_direction]
        elif self.mode == "keydoor":
            message = "KEY HELD" if self.has_key else "FIND KEY"
        elif self.mode == "rhythm":
            message = "GATE OPEN" if self.steps % self.beat_period == 0 else "GATE CLOSED"
        else:
            message = ""
        encoded = np.zeros(characters, dtype=np.int64)
        raw = np.frombuffer(message.encode("ascii"), dtype=np.uint8)
        encoded[:min(characters, len(raw))] = raw[:characters]
        return encoded


def make_game(name: str, size: int, seed: int) -> PrimitiveGame:
    if name == "snake":
        return SnakeAdapter(size=size, seed=seed)
    constructors = {"collect": CollectEnv, "dodge": DodgeEnv, "chase": ChaseEnv}
    if name in MissionEnv.MODES:
        return MissionEnv(name, size=size, seed=seed)
    if name not in constructors:
        raise ValueError(f"unknown game {name!r}; choose from {', '.join(GAME_NAMES)}")
    return constructors[name](size=size, seed=seed)


class SnakeAdapter:
    """Expose Snake through the suite's generic naming contract."""

    def __init__(self, size: int, seed: int):
        self.env = SnakeEnv(size=size, seed=seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observe(self, target_visible: bool = True, detail_visible: bool = True,
                theme: int = 0) -> np.ndarray:
        return self.env.observe(target_visible, detail_visible, theme)

    def raw_audio(self, samples: int = 64) -> np.ndarray:
        if not self.env.state.ate_last:
            return np.zeros(samples, dtype=np.float32)
        phase = np.linspace(0, 8 * np.pi, samples, endpoint=False)
        return (0.5 * np.sin(phase)).astype(np.float32)

    def raw_text(self, characters: int = 32) -> np.ndarray:
        return np.zeros(characters, dtype=np.int64)


def make_multigame_dataset(samples: int, sequence: int, size: int, seed: int,
                           games: tuple[str, ...], partial: bool = True,
                           themes: tuple[int, ...] = (0,)) -> dict[str, np.ndarray]:
    if not games:
        raise ValueError("at least one game is required")
    rng = np.random.default_rng(seed)
    envs = {name: make_game(name, size, seed + 101 * index)
            for index, name in enumerate(games)}
    frames, audio, text, semantic, game_ids = [], [], [], [], []
    labels: dict[str, list] = {key: [] for key in
                               ("action", "horizontal", "vertical", "direction", "danger", "ate")}
    game_index = 0
    per_episode = max(1, min(16, samples // (len(games) * 4)))
    while len(frames) < samples:
        name = games[game_index % len(games)]
        game_index += 1
        env = envs[name]
        env.reset()
        first_theme = themes[int(rng.integers(len(themes)))]
        first = env.observe(True, True, first_theme)
        first_audio = env.raw_audio()
        first_text = env.raw_text()
        history: deque[np.ndarray] = deque((first.copy() for _ in range(sequence)), maxlen=sequence)
        audio_history: deque[np.ndarray] = deque(
            (first_audio.copy() for _ in range(sequence)), maxlen=sequence)
        text_history: deque[np.ndarray] = deque(
            (first_text.copy() for _ in range(sequence)), maxlen=sequence)
        episode_examples = 0
        for _ in range(300):
            visible = not partial or rng.random() > 0.34
            detail = not partial or rng.random() > 0.18
            theme = themes[int(rng.integers(len(themes)))]
            history.append(env.observe(visible, detail, theme))
            audio_history.append(env.raw_audio())
            text_history.append(env.raw_text())
            if len(history) == sequence:
                frames.append(np.stack(history))
                audio.append(np.stack(audio_history))
                text.append(np.stack(text_history))
                semantic.append(env.semantic_vector())
                game_ids.append(GAME_NAMES.index(name))
                episode_examples += 1
                current = env.labels()
                for key in labels:
                    labels[key].append(current[key])
                if len(frames) >= samples:
                    break
                if episode_examples >= per_episode:
                    break
            _, done = env.step(env.teacher_action())
            if done:
                break
    result = {
        "frames": np.stack(frames),
        "audio": np.stack(audio),
        "text": np.stack(text),
        "semantic": np.stack(semantic),
        "game": np.asarray(game_ids, dtype=np.int64),
    }
    result.update({key: np.asarray(value) for key, value in labels.items()})
    return result
