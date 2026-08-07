"""Parameterized game family for compositional fragment induction (R1).

One verifier-private grid world whose variants are built from four shared
components — collection, interception, avoidance, navigation — each with
parameter levels. A variant is a component subset plus settings. The family
provides the compositional support the fragment bank needs: every component
value appears in some training combination, and held-out combinations exist
for novel-recombination probes.

The avatar moves in four directions on an 8x8 grid. Rendering uses three
planes compatible with the shared screen driver: avatar, positive objects
(food, falling catchables, goal), negative objects (hazards, walls).
Rewards: +1 collect/catch/reach-goal, -1 hazard contact or missed catch
(both end the row's episode). All state, rules, and component identities
stay verifier-private.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import torch

from experiments.games_amodal.environments import GameStep

FAMILY_ACTION_COUNT = 4
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

COMPONENTS = ("collect", "intercept", "avoid", "navigate")


@dataclass(frozen=True)
class FamilyConfig:
    """One variant: which components are active and at what level."""

    collect: int = 0  # number of food items (0 = off)
    intercept: int = 0  # number of simultaneous falling objects
    avoid: int = 0  # number of moving hazards
    navigate: bool = False  # goal tile behind wall segments
    forage: int = 0  # pairs of two visually distinct item types; one type
    # is food (+1), the other fatal (-1). `inverted` swaps which is which,
    # so twins render identically and only context can say which to eat.
    # Passivity scores zero: mastery requires eating the right type.
    inverted: bool = False  # SAME rendering, opposite meaning: touching a
    # positive-plane object (food/goal) is -1 and fatal. Observation alone
    # cannot reveal the objective; only fetched context can.
    name: str = field(default="", compare=False)

    def active(self) -> tuple[str, ...]:
        return tuple(
            component
            for component, level in (
                ("collect", self.collect),
                ("intercept", self.intercept),
                ("avoid", self.avoid),
                ("navigate", int(self.navigate)),
                ("forage", self.forage),
            )
            if level
        )

    def validate(self) -> FamilyConfig:
        if not self.active():
            raise ValueError("a variant needs at least one active component")
        if min(self.collect, self.intercept, self.avoid, self.forage) < 0:
            raise ValueError("component levels cannot be negative")
        if max(self.collect, self.intercept, self.avoid, self.forage) > 3:
            raise ValueError("component levels above 3 are not supported")
        return self


class FamilyVerifier:
    """Batched composigrid verifier for one variant configuration."""

    action_count = FAMILY_ACTION_COUNT

    def __init__(
        self,
        config: FamilyConfig,
        *,
        batch_size: int,
        height: int = 8,
        width: int = 8,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        config.validate()
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if min(height, width) < 6:
            raise ValueError("grid must be at least 6x6")
        self.config = config
        self.batch_size = int(batch_size)
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        self._avatar: list[tuple[int, int]] = []
        self._food: list[list[tuple[int, int]]] = []
        self._fallers: list[list[tuple[int, int]]] = []
        self._hazards: list[list[tuple[int, int, int]]] = []
        self._walls: list[set[tuple[int, int]]] = []
        self._goal: list[tuple[int, int] | None] = []
        self._forage_a: list[list[tuple[int, int]]] = []
        self._forage_b: list[list[tuple[int, int]]] = []
        self._alive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    def _rand(self, high: int) -> int:
        return int(
            torch.randint(
                0, high, (1,), generator=self._generator, device=self.device
            ).item()
        )

    def _free_cell(self, row: int, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            cell = (self._rand(self.height), self._rand(self.width))
            if cell not in occupied:
                return cell

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._generator.manual_seed(int(seed))
        config = self.config
        self._avatar = []
        self._food = []
        self._fallers = []
        self._hazards = []
        self._walls = []
        self._goal = []
        self._alive = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        for row in range(self.batch_size):
            occupied: set[tuple[int, int]] = set()
            walls: set[tuple[int, int]] = set()
            if config.navigate:
                wall_column = 2 + self._rand(self.width - 4)
                gap = self._rand(self.height)
                for line in range(self.height):
                    if line != gap:
                        walls.add((line, wall_column))
                occupied |= walls
            avatar = self._free_cell(row, occupied)
            occupied.add(avatar)
            self._avatar.append(avatar)
            self._walls.append(walls)
            food = []
            for _ in range(config.collect):
                cell = self._free_cell(row, occupied)
                occupied.add(cell)
                food.append(cell)
            self._food.append(food)
            fallers = []
            for _ in range(config.intercept):
                fallers.append((0, self._rand(self.width)))
            self._fallers.append(fallers)
            hazards = []
            for _ in range(config.avoid):
                cell = self._free_cell(row, occupied)
                occupied.add(cell)
                hazards.append((cell[0], cell[1], 1 if self._rand(2) else -1))
            self._hazards.append(hazards)
            if config.navigate:
                goal = self._free_cell(row, occupied)
                occupied.add(goal)
                self._goal.append(goal)
            else:
                self._goal.append(None)
            type_a, type_b = [], []
            for _ in range(config.forage):
                cell = self._free_cell(row, occupied)
                occupied.add(cell)
                type_a.append(cell)
                cell = self._free_cell(row, occupied)
                occupied.add(cell)
                type_b.append(cell)
            self._forage_a.append(type_a)
            self._forage_b.append(type_b)

    def observation(self) -> torch.Tensor:
        """Render [batch, 3, h, w]: avatar, positive objects, negatives."""

        grid = torch.zeros(
            self.batch_size, 3, self.height, self.width, device=self.device
        )
        for row in range(self.batch_size):
            if not bool(self._alive[row]):
                continue
            avatar = self._avatar[row]
            grid[row, 0, avatar[0], avatar[1]] = 1.0
            for cell in self._food[row]:
                grid[row, 1, cell[0], cell[1]] = 1.0
            for cell in self._fallers[row]:
                grid[row, 1, cell[0], cell[1]] = 1.0
            goal = self._goal[row]
            if goal is not None:
                grid[row, 1, goal[0], goal[1]] = 1.0
            for hazard in self._hazards[row]:
                grid[row, 2, hazard[0], hazard[1]] = 1.0
            for cell in self._walls[row]:
                grid[row, 2, cell[0], cell[1]] = 1.0
            for cell in self._forage_a[row]:
                grid[row, 1, cell[0], cell[1]] = 1.0
            for cell in self._forage_b[row]:
                grid[row, 2, cell[0], cell[1]] = 1.0
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
            delta = _DELTAS[int(actions[row].item())]
            avatar = self._avatar[row]
            target = (avatar[0] + delta[0], avatar[1] + delta[1])
            if (
                not (0 <= target[0] < self.height and 0 <= target[1] < self.width)
                or target in self._walls[row]
            ):
                target = avatar
            self._avatar[row] = target

            if target in self._food[row]:
                self._food[row].remove(target)
                if self.config.inverted:
                    reward[row] -= 1.0
                    self._alive[row] = False
                    continue
                reward[row] += 1.0
                occupied = set(self._food[row]) | self._walls[row] | {target}
                self._food[row].append(self._free_cell(row, occupied))

            goal = self._goal[row]
            if goal is not None and target == goal:
                if self.config.inverted:
                    reward[row] -= 1.0
                    self._alive[row] = False
                    continue
                reward[row] += 1.0
                occupied = set(self._food[row]) | self._walls[row] | {target}
                self._goal[row] = self._free_cell(row, occupied)

            good, bad = self._forage_a[row], self._forage_b[row]
            if self.config.inverted:
                good, bad = bad, good
            if target in good:
                good.remove(target)
                reward[row] += 1.0
                occupied = (
                    set(good) | set(bad) | self._walls[row] | {target}
                )
                good.append(self._free_cell(row, occupied))
            elif target in bad:
                reward[row] -= 1.0
                self._alive[row] = False
                continue

            next_fallers = []
            for faller in self._fallers[row]:
                dropped = (faller[0] + 1, faller[1])
                if dropped[0] >= self.height:
                    if (self.height - 1, dropped[1]) == target:
                        reward[row] += 1.0
                    else:
                        reward[row] -= 1.0
                        self._alive[row] = False
                    next_fallers.append((0, self._rand(self.width)))
                elif dropped == target:
                    reward[row] += 1.0
                    next_fallers.append((0, self._rand(self.width)))
                else:
                    next_fallers.append(dropped)
            self._fallers[row] = next_fallers
            if not bool(self._alive[row]):
                continue

            next_hazards = []
            hit = False
            for hazard in self._hazards[row]:
                position = (hazard[0], hazard[1] + hazard[2])
                direction = hazard[2]
                if position[1] < 0 or position[1] >= self.width:
                    direction = -direction
                    position = (hazard[0], hazard[1] + direction)
                if position == target:
                    hit = True
                next_hazards.append((position[0], position[1], direction))
            self._hazards[row] = next_hazards
            if hit:
                reward[row] -= 1.0
                self._alive[row] = False
        return GameStep(reward=reward, alive=self._alive.clone())


def family_variants(
    *,
    max_components: int = 2,
    levels: tuple[int, ...] = (1, 2),
) -> list[FamilyConfig]:
    """Enumerate the variant family over component subsets and levels."""

    variants: list[FamilyConfig] = []
    for size in range(1, max_components + 1):
        for subset in combinations(COMPONENTS, size):
            leveled = [
                levels if component != "navigate" else (1,)
                for component in subset
            ]
            for chosen in product(*leveled):
                settings = dict(zip(subset, chosen, strict=True))
                config = FamilyConfig(
                    collect=settings.get("collect", 0),
                    intercept=settings.get("intercept", 0),
                    avoid=settings.get("avoid", 0),
                    navigate=bool(settings.get("navigate", 0)),
                    name="+".join(
                        f"{component}{settings[component]}"
                        for component in subset
                    ),
                )
                variants.append(config)
    return variants


def compositional_split(
    variants: list[FamilyConfig], *, holdout_fraction: float = 0.25, seed: int = 0
) -> tuple[list[FamilyConfig], list[FamilyConfig]]:
    """Split so held-out variants are novel COMBINATIONS of seen components.

    Every component at every level must appear in at least one training
    variant (compositional support); only multi-component variants are
    eligible for holdout.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout fraction must lie in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    eligible = [v for v in variants if len(v.active()) > 1]
    order = torch.randperm(len(eligible), generator=generator).tolist()
    target = max(1, int(len(eligible) * holdout_fraction))
    holdout: list[FamilyConfig] = []
    for index in order:
        candidate = eligible[index]
        remaining = [
            v for v in variants if v not in holdout and v != candidate
        ]
        support = {
            (component, getattr(v, component))
            for v in remaining
            for component in COMPONENTS
            if getattr(v, component)
        }
        needed = {
            (component, getattr(candidate, component))
            for component in COMPONENTS
            if getattr(candidate, component)
        }
        if needed <= support:
            holdout.append(candidate)
        if len(holdout) >= target:
            break
    train = [v for v in variants if v not in holdout]
    return train, holdout
