"""Vectorized family verifier: torch-native batched dynamics.

A performance re-implementation of `FamilyVerifier` for the component
subset the current probes exercise: collect, intercept, avoid, pursue,
delayed, resource, deceptive (bait). Everything is tensor state --
no per-row Python loops in step() -- so wall-clock scales with tensor
ops, not batch size, and the whole verifier moves to any torch device.

Semantics follow game_family.py's step() ordering: move -> food ->
delayed switch -> resource -> bait -> pursuers -> fallers -> hazards.
Random respawns use bounded rejection sampling (32 rounds) instead of
loop-until-free; on an 8x8 grid with a handful of objects this leaves
the distribution indistinguishable in practice, but runs are NOT
bit-comparable with the reference implementation -- fast-verifier
results pair only against fast-verifier results. Behavioural
equivalence is pinned by tests/test_fast_family.py, which replays the
reference implementation's semantic unit tests against this class.

Unsupported components (navigate/walls, forage, choice, dual, blink,
oneway, lever, faller knobs) raise at construction: the sealed trio
stays out of the fast path on purpose.
"""

from __future__ import annotations

import torch

from experiments.games_amodal.environments import GameStep
from experiments.games_amodal.game_family import FamilyConfig

_DELTAS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])


class FastFamilyVerifier:
    action_count = 4

    def __init__(self, config: FamilyConfig, *, batch_size: int,
                 height: int = 8, width: int = 8, seed: int = 0,
                 device: torch.device | str = "cpu") -> None:
        config.validate()
        unsupported = {
            "navigate": config.navigate, "forage": config.forage,
            "choice": config.choice, "dual": config.dual,
            "blink": config.blink, "oneway": config.oneway,
            "lever": config.lever, "inverted": config.inverted,
            "faller_spread": config.faller_spread,
            "recentre": config.recentre_every,
            "spawn_radius": config.spawn_radius,
            "slow_fallers": config.faller_period != 1,
        }
        bad = [name for name, level in unsupported.items() if level]
        if bad:
            raise ValueError(f"fast verifier does not support: {bad}")
        self.config = config
        self.batch_size = int(batch_size)
        self.height, self.width = int(height), int(width)
        self.device = torch.device(device)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self._deltas = _DELTAS.to(self.device)
        self._step_index = 0

    # -- helpers ---------------------------------------------------------

    def _rand_cells(self, n: int) -> torch.Tensor:
        r = torch.randint(0, self.height, (n,), generator=self._gen)
        c = torch.randint(0, self.width, (n,), generator=self._gen)
        return torch.stack([r, c], dim=1).to(self.device)

    def _occupied_mask(self) -> torch.Tensor:
        """[B, H*W] bool of every occupied cell."""
        occ = torch.zeros(self.batch_size, self.height * self.width,
                          dtype=torch.bool, device=self.device)

        def mark(pos, present=None):
            flat = pos[..., 0] * self.width + pos[..., 1]
            if flat.dim() == 1:
                occ.scatter_(1, flat.unsqueeze(1).clamp(min=0), True)
            else:
                safe = flat.clamp(min=0)
                val = (present if present is not None
                       else torch.ones_like(safe, dtype=torch.bool))
                occ.scatter_(1, safe, occ.gather(1, safe) | val)

        mark(self.avatar)
        for name in ("food", "resources", "bait", "pursuers"):
            t = getattr(self, name)
            if t.shape[1]:
                mark(t, getattr(self, name + "_present", None))
        if self.hazards.shape[1]:
            mark(self.hazards[..., :2])
        if self.config.delayed:
            mark(self.switch)
        return occ

    def _respawn(self, rows: torch.Tensor) -> torch.Tensor:
        """Free cells for `rows` (bool [B]) via bounded rejection."""
        n = int(rows.sum())
        cells = self._rand_cells(n)
        occ = self._occupied_mask()[rows]
        for _ in range(32):
            flat = cells[:, 0] * self.width + cells[:, 1]
            bad = occ.gather(1, flat.unsqueeze(1)).squeeze(1)
            if not bool(bad.any()):
                break
            fresh = self._rand_cells(int(bad.sum()))
            cells = cells.clone()
            cells[bad] = fresh
        return cells

    # -- lifecycle -------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._gen.manual_seed(int(seed))
        B, cfg = self.batch_size, self.config
        self._step_index = 0
        self.alive = torch.ones(B, dtype=torch.bool, device=self.device)
        placed = torch.zeros(B, self.height * self.width, dtype=torch.bool,
                             device=self.device)

        def place(n):
            """n non-colliding cells per row, sequentially rejected."""
            out = torch.full((B, n, 2), -1, dtype=torch.long,
                             device=self.device)
            for k in range(n):
                cells = self._rand_cells(B)
                for _ in range(32):
                    flat = cells[:, 0] * self.width + cells[:, 1]
                    bad = placed.gather(1, flat.unsqueeze(1)).squeeze(1)
                    if not bool(bad.any()):
                        break
                    fresh = self._rand_cells(int(bad.sum()))
                    cells = cells.clone()
                    cells[bad] = fresh
                flat = cells[:, 0] * self.width + cells[:, 1]
                placed.scatter_(1, flat.unsqueeze(1), True)
                out[:, k] = cells
            return out

        self.avatar = place(1)[:, 0]
        self.food = place(cfg.collect)
        self.hazards = torch.cat([
            place(cfg.avoid),
            (torch.randint(0, 2, (B, cfg.avoid, 1), generator=self._gen)
             .to(self.device) * 2 - 1)], dim=2) if cfg.avoid else \
            torch.zeros(B, 0, 3, dtype=torch.long, device=self.device)
        self.pursuers = place(cfg.pursue)
        self.resources = place(cfg.resource)
        self.holding = torch.zeros(B, dtype=torch.long, device=self.device)
        if cfg.delayed:
            self.switch = place(1)[:, 0]
        else:
            self.switch = torch.full((B, 2), -1, dtype=torch.long,
                                     device=self.device)
        self.pending = torch.zeros(B, 0, dtype=torch.long,
                                   device=self.device)
        if cfg.deceptive:
            self.bait = torch.stack(
                [self._bait_cells() for _ in range(cfg.deceptive)], dim=1)
        else:
            self.bait = torch.zeros(B, 0, 2, dtype=torch.long,
                                    device=self.device)
        self.fallers = torch.stack([
            torch.zeros(B, cfg.intercept, dtype=torch.long,
                        device=self.device),
            torch.randint(0, self.width, (B, cfg.intercept),
                          generator=self._gen).to(self.device)], dim=2) \
            if cfg.intercept else torch.zeros(B, 0, 2, dtype=torch.long,
                                              device=self.device)

    def _bait_cells(self) -> torch.Tensor:
        """One cell per row adjacent (Chebyshev 1) to a random hazard."""
        B = self.batch_size
        idx = torch.randint(0, max(1, self.hazards.shape[1]), (B,),
                            generator=self._gen).to(self.device)
        anchor = self.hazards[torch.arange(B, device=self.device),
                              idx, :2]
        off = torch.randint(-1, 2, (B, 2), generator=self._gen).to(
            self.device)
        zero = (off == 0).all(dim=1)
        off[zero, 0] = 1
        cells = (anchor + off).clamp(
            min=torch.zeros(2, dtype=torch.long, device=self.device),
            max=torch.tensor([self.height - 1, self.width - 1],
                             device=self.device))
        on_anchor = (cells == anchor).all(dim=1)
        flipped = (anchor - off).clamp(
            min=torch.zeros(2, dtype=torch.long, device=self.device),
            max=torch.tensor([self.height - 1, self.width - 1],
                             device=self.device))
        cells = torch.where(on_anchor.unsqueeze(1), flipped, cells)
        return cells

    def observation(self) -> torch.Tensor:
        B = self.batch_size
        grid = torch.zeros(B, 3, self.height, self.width,
                           device=self.device)
        rows = torch.arange(B, device=self.device)

        def draw(plane, pos, present=None):
            if pos.numel() == 0:
                return
            if pos.dim() == 2:
                pos = pos.unsqueeze(1)
            n = pos.shape[1]
            keep = (present if present is not None else
                    torch.ones(B, n, dtype=torch.bool, device=self.device))
            keep = keep & self.alive.unsqueeze(1)
            r = pos[..., 0].clamp(0, self.height - 1)
            c = pos[..., 1].clamp(0, self.width - 1)
            flat = (plane * self.height * self.width + r * self.width + c)
            # presence only: overlapping draws stay 1.0, so plain
            # scatter of {0,1} is equivalent to add-then-binarize
            marks = torch.where(keep, torch.ones_like(flat).float(),
                                torch.zeros_like(flat).float())
            view = grid.view(B, -1)
            view.scatter_(1, flat, torch.maximum(
                view.gather(1, flat), marks))

        draw(0, self.avatar)
        draw(1, self.food)
        draw(1, self.bait)
        draw(1, self.fallers)
        draw(2, self.hazards[..., :2])
        draw(2, self.pursuers)
        draw(2, self.resources)
        if self.config.delayed:
            draw(2, self.switch)
        return grid

    # -- dynamics --------------------------------------------------------

    def step(self, actions: torch.Tensor) -> GameStep:
        B, cfg = self.batch_size, self.config
        actions = actions.to(self.device)
        reward = torch.zeros(B, device=self.device)
        alive = self.alive
        delta = self._deltas[actions]
        target = self.avatar + delta
        inside = ((target[:, 0] >= 0) & (target[:, 0] < self.height)
                  & (target[:, 1] >= 0) & (target[:, 1] < self.width))
        target = torch.where((inside & alive).unsqueeze(1), target,
                             self.avatar)
        self.avatar = target

        def touched(pos):
            if pos.shape[1] == 0:
                return torch.zeros(B, 0, dtype=torch.bool,
                                   device=self.device)
            return ((pos == target.unsqueeze(1)).all(dim=2)
                    & alive.unsqueeze(1))

        # food
        hit = touched(self.food)
        if hit.shape[1]:
            got = hit.any(dim=1)
            if cfg.resource:
                fueled = got & (self.holding > 0)
                reward = reward + fueled.float()
                self.holding = self.holding - fueled.long()
            else:
                reward = reward + got.float()
            if bool(got.any()):
                fresh = self._respawn(got)
                idx = hit.float().argmax(dim=1)
                food = self.food.clone()
                food[got, idx[got]] = fresh
                self.food = food

        # delayed switch: pending slot 0 = empty, >=1 = active timer
        if cfg.delayed:
            if self.pending.shape[1]:
                active = self.pending > 0
                self.pending = torch.where(active, self.pending - 1,
                                           self.pending)
                due = active & (self.pending == 0) & alive.unsqueeze(1)
                reward = reward + due.float().sum(dim=1)
            pressed = (self.switch == target).all(dim=1) & alive
            if bool(pressed.any()):
                column = torch.where(
                    pressed,
                    torch.full((B,), cfg.delayed, device=self.device),
                    torch.zeros(B, dtype=torch.long, device=self.device))
                self.pending = torch.cat(
                    [self.pending, column.unsqueeze(1)], dim=1)
                fresh = self._respawn(pressed)
                switch = self.switch.clone()
                switch[pressed] = fresh
                self.switch = switch

        # resources
        hit = touched(self.resources)
        if hit.shape[1]:
            got = hit.any(dim=1)
            self.holding = self.holding + got.long()
            if bool(got.any()):
                fresh = self._respawn(got)
                idx = hit.float().argmax(dim=1)
                res = self.resources.clone()
                res[got, idx[got]] = fresh
                self.resources = res

        # bait
        hit = touched(self.bait)
        if hit.shape[1]:
            got = hit.any(dim=1)
            reward = reward + 0.2 * got.float()
            if bool(got.any()):
                fresh = self._bait_cells()
                idx = hit.float().argmax(dim=1)
                bait = self.bait.clone()
                bait[got, idx[got]] = fresh
                self.bait = bait

        # pursuers
        if self.pursuers.shape[1]:
            gap = target.unsqueeze(1) - self.pursuers
            move_row = (gap[..., 0].abs() >= gap[..., 1].abs()) \
                & (gap[..., 0] != 0)
            move_col = ~move_row & (gap[..., 1] != 0)
            step_r = torch.where(move_row, gap[..., 0].sign(),
                                 torch.zeros_like(gap[..., 0]))
            step_c = torch.where(move_col, gap[..., 1].sign(),
                                 torch.zeros_like(gap[..., 1]))
            self.pursuers = self.pursuers + torch.stack(
                [step_r, step_c], dim=2)
            caught = ((self.pursuers == target.unsqueeze(1)).all(dim=2)
                      .any(dim=1) & alive)
            reward = reward - caught.float()
            alive = alive & ~caught

        # fallers
        if self.fallers.shape[1]:
            dropped = self.fallers.clone()
            dropped[..., 0] += 1
            landed = dropped[..., 0] >= self.height
            at_catch = (dropped == target.unsqueeze(1)).all(dim=2)
            floor = torch.stack(
                [torch.full_like(dropped[..., 0], self.height - 1),
                 dropped[..., 1]], dim=2)
            floor_catch = landed & (floor == target.unsqueeze(1)).all(dim=2)
            caught = (at_catch & ~landed) | floor_catch
            missed = landed & ~floor_catch
            reward = reward + (caught.float().sum(dim=1)
                               - (missed & alive.unsqueeze(1)).float()
                               .sum(dim=1)) * alive.float()
            died = missed.any(dim=1) & alive
            alive = alive & ~died
            renew = caught | landed
            fresh_cols = torch.randint(
                0, self.width, (B, self.fallers.shape[1]),
                generator=self._gen).to(self.device)
            dropped[..., 0] = torch.where(
                renew, torch.zeros_like(dropped[..., 0]), dropped[..., 0])
            dropped[..., 1] = torch.where(renew, fresh_cols,
                                          dropped[..., 1])
            self.fallers = dropped

        # hazards
        if self.hazards.shape[1]:
            pos_c = self.hazards[..., 1] + self.hazards[..., 2]
            flip = (pos_c < 0) | (pos_c >= self.width)
            direction = torch.where(flip, -self.hazards[..., 2],
                                    self.hazards[..., 2])
            pos_c = torch.where(flip,
                                self.hazards[..., 1] + direction, pos_c)
            self.hazards = torch.stack(
                [self.hazards[..., 0], pos_c, direction], dim=2)
            hit = ((self.hazards[..., :2] == target.unsqueeze(1))
                   .all(dim=2).any(dim=1) & alive)
            reward = reward - hit.float()
            alive = alive & ~hit

        reward = torch.where(self.alive, reward,
                             torch.zeros_like(reward))
        self.alive = alive
        self._step_index += 1
        return GameStep(reward=reward, alive=self.alive.clone())
