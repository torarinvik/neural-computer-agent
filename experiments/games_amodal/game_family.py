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

# `dual` renders the two choices identically in both trial kinds (one on
# each object plane) and signals WHICH RULE IS IN FORCE with a separate
# cue on the avatar plane: the top row's left half (kind 0) or right half
# (kind 1). Encoding the trial kind into the items themselves instead --
# as a second intensity level -- makes each rule a plane-x-intensity
# conjunction this plant cannot learn even with a single unambiguous
# context to itself (probe 17). A single cue *pixel* is legible but too
# faint against the two-item choice: conditional contexts plateau at 0.75
# (probe 18). The cue must carry salience comparable to the choice.

# F2, strengthened for multi-rule worlds. An agent that has mastered one
# trial kind SELECTIVELY refuses the kind it has not mastered, and its
# per-rule accuracy on the refused kind then measures nothing. Two things
# are needed to close the escape, and the first alone is not enough:
#
#   1. Idling must cost something (a uniform per-step cost does not work;
#      it is paid whether or not the agent engages).
#   2. Guessing must PAY. Under policy gradient a symmetric +1/-1 coin
#      flip is a zero-mean, high-variance action, and the low-variance
#      idle action wins even though it is worse in expectation. The agent
#      then never gathers the trials it needs to learn the rule at all.
#
# So engaging under ignorance earns +0.4 in expectation, idling earns
# -0.1, and knowing the rule earns +1.0. Each step of competence strictly
# pays for itself.
DUAL_IDLE_COST = 0.1
DUAL_WRONG_COST = 0.2


@dataclass(frozen=True)
class FamilyConfig:
    """One variant: which components are active and at what level."""

    collect: int = 0  # number of food items (0 = off)
    intercept: int = 0  # number of simultaneous falling objects
    avoid: int = 0  # number of moving hazards
    navigate: bool = False  # goal tile behind wall segments
    forage: int = 0  # pairs of two visually distinct item types; one type
    # is food (+1), the other costly (-1) but NOT fatal — both respawn.
    # `inverted` swaps which is which, so twins render identically and only
    # context can say which to eat. Passivity scores zero, so mastery
    # requires net-positive eating; survivable mistakes keep a gradient.
    choice: int = 0  # minimal disambiguation trial: the avatar sits at the
    # centre with one type-A item and one type-B item on adjacent cells.
    # Stepping onto one resolves the trial (+1 / -1 by type, `inverted`
    # swaps which), then a fresh pair appears. Navigation is one step, so
    # the only thing to learn is WHICH TYPE to eat — the pure bank test.
    inverted: bool = False  # SAME rendering, opposite meaning: touching a
    # positive-plane object (food/goal) is -1 and fatal. Observation alone
    # cannot reveal the objective; only fetched context can.
    dual: int = 0  # TWO independent binary rules in one world. Each trial
    # offers the same two choices — one item on each object plane — and a
    # separate cue cell says which of two rules is in force. `inverted`
    # sets the rule for cue-0 trials (take side 0 = A, or side 1 = B);
    # `inverted2` sets it for cue-1 trials (C or D). The cue is visible,
    # the rule it selects is not. Contexts are therefore a product of two
    # bits, so contexts sharing a bit share a sub-rule -- the structure a
    # fragment bank must reuse rather than re-learn.
    inverted2: bool = False  # second axis of `dual` (rule for cue-1 trials)
    spawn_radius: int = 0  # F19 motor bridge: 0 = items spawn anywhere;
    # r>0 = forage items spawn within Chebyshev radius r of the avatar.
    # Radius 1 makes forage the already-mastered choice trial; growing r
    # is a curriculum that lets the decision skill seed the navigation
    # skill the plant cannot bootstrap cold (F5 staging, applied inside
    # one game).
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
                ("choice", self.choice),
                ("dual", self.dual),
            )
            if level
        )

    def rules(self) -> tuple[str, ...]:
        """The independent sub-rules this variant obeys.

        Two variants that share a rule share learnable content; a bank that
        factorizes should give them a common fragment. Verifier-side
        bookkeeping for the harness only -- never shown to the learner.
        """

        if not self.dual:
            return ()
        return (
            "takeB" if self.inverted else "takeA",
            "takeD" if self.inverted2 else "takeC",
        )

    def validate(self) -> FamilyConfig:
        if not self.active():
            raise ValueError("a variant needs at least one active component")
        levels = (
            self.collect,
            self.intercept,
            self.avoid,
            self.forage,
            self.choice,
            self.dual,
        )
        if min(levels) < 0:
            raise ValueError("component levels cannot be negative")
        if self.spawn_radius < 0:
            raise ValueError("spawn radius cannot be negative")
        if self.spawn_radius and not self.forage:
            raise ValueError("spawn radius is a forage curriculum knob")
        if max(levels) > 3:
            raise ValueError("component levels above 3 are not supported")
        if self.dual and self.choice:
            raise ValueError("dual and choice both own the trial cycle")
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
        self._dual_items: list[list[tuple[int, int, int]]] = []
        self._dual_kind: list[int] = []
        self._dual_stats = torch.zeros(2, 2, device=self.device)
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

    def _forage_cell(self, row: int, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        radius = self.config.spawn_radius
        if radius <= 0:
            return self._free_cell(row, occupied)
        centre = self._avatar[row]
        candidates = [
            (r, c)
            for r in range(max(0, centre[0] - radius), min(self.height, centre[0] + radius + 1))
            for c in range(max(0, centre[1] - radius), min(self.width, centre[1] + radius + 1))
            if (r, c) != centre and (r, c) not in occupied
        ]
        if not candidates:
            return self._free_cell(row, occupied)
        return candidates[self._rand(len(candidates))]

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
        self._forage_a = []
        self._forage_b = []
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
                cell = self._forage_cell(row, occupied)
                occupied.add(cell)
                type_a.append(cell)
                cell = self._forage_cell(row, occupied)
                occupied.add(cell)
                type_b.append(cell)
            self._forage_a.append(type_a)
            self._forage_b.append(type_b)
            if config.choice:
                self._avatar[row] = (self.height // 2, self.width // 2)
                self._deal_choice(row)
        self._dual_items = [[] for _ in range(self.batch_size)]
        self._dual_kind = [0 for _ in range(self.batch_size)]
        self._dual_stats = torch.zeros(2, 2, device=self.device)
        if config.dual:
            for row in range(self.batch_size):
                self._deal_dual(row)

    def _neighbour_pair(self, row: int) -> tuple[tuple[int, int], tuple[int, int]]:
        centre = (self.height // 2, self.width // 2)
        self._avatar[row] = centre
        neighbours = [
            (centre[0] + delta[0], centre[1] + delta[1]) for delta in _DELTAS
        ]
        first = self._rand(len(neighbours))
        second = (first + 1 + self._rand(len(neighbours) - 1)) % len(neighbours)
        return neighbours[first], neighbours[second]

    def _deal_dual(self, row: int) -> None:
        """Deal one trial of a randomly chosen kind, cue and choices apart."""

        left, right = self._neighbour_pair(row)
        self._dual_kind[row] = self._rand(2)
        self._dual_items[row] = [
            (left[0], left[1], 0),
            (right[0], right[1], 1),
        ]

    def dual_edible_side(self, kind: int) -> int:
        flipped = self.config.inverted2 if kind else self.config.inverted
        return 1 if flipped else 0

    def dual_accuracy(self) -> list[float]:
        """Per-axis fraction of trials resolved correctly (harness only)."""

        totals = self._dual_stats[:, 1].clamp_min(1.0)
        return (self._dual_stats[:, 0] / totals).tolist()

    def dual_engagement(self) -> list[float]:
        """Trials resolved per axis (harness only).

        Accuracy alone cannot distinguish "has not learned this rule" from
        "declines to play this trial kind": both read ~0.5. An agent that
        knows one rule can leave the other trial kind unresolved, paying
        only an opportunity cost instead of the -1 it would earn by
        guessing. Engagement is how that evasion becomes visible.
        """

        return self._dual_stats[:, 1].tolist()

    def _deal_choice(self, row: int) -> None:
        """Place one type-A and one type-B item adjacent to the avatar."""

        left, right = self._neighbour_pair(row)
        self._forage_a[row] = [left]
        self._forage_b[row] = [right]

    def observation(self) -> torch.Tensor:
        """Render [batch, 3, h, w]: avatar, positive objects, negatives.

        `dual` item classes are drawn as four distinguishable marks inside
        the same three planes: A = 1.0 positive, B = 1.0 negative,
        C = 0.5 positive, D = 0.5 negative. The mark says which trial kind
        is on screen; it never says which mark is edible.
        """

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
            if self.config.dual and self._dual_items[row]:
                half = self.width // 2
                if self._dual_kind[row] == 0:
                    grid[row, 0, 0, :half] = 1.0
                else:
                    grid[row, 0, 0, half:] = 1.0
                for item in self._dual_items[row]:
                    grid[row, 1 + item[2], item[0], item[1]] = 1.0
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

            if self.config.dual:
                kind = self._dual_kind[row]
                resolved = False
                for item in self._dual_items[row]:
                    if (item[0], item[1]) != target:
                        continue
                    correct = item[2] == self.dual_edible_side(kind)
                    reward[row] += 1.0 if correct else -DUAL_WRONG_COST
                    self._dual_stats[kind, 0] += float(correct)
                    self._dual_stats[kind, 1] += 1.0
                    self._deal_dual(row)
                    resolved = True
                    break
                if not resolved:
                    reward[row] -= DUAL_IDLE_COST
                # Always recentre: the avatar can never wander off the
                # decision, so the trial is a pure repeated two-alternative
                # choice (F7) and the cue corners stay unoccupied.
                self._avatar[row] = (self.height // 2, self.width // 2)
                continue

            good, bad = self._forage_a[row], self._forage_b[row]
            if self.config.inverted:
                good, bad = bad, good
            if self.config.choice:
                if target in good:
                    reward[row] += 1.0
                    self._deal_choice(row)
                elif target in bad:
                    reward[row] -= 1.0
                    self._deal_choice(row)
                continue
            if target in good:
                good.remove(target)
                reward[row] += 1.0
                occupied = (
                    set(good) | set(bad) | self._walls[row] | {target}
                )
                good.append(self._forage_cell(row, occupied))
            elif target in bad:
                bad.remove(target)
                reward[row] -= 1.0
                occupied = (
                    set(good) | set(bad) | self._walls[row] | {target}
                )
                bad.append(self._forage_cell(row, occupied))

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
