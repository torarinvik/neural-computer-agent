"""Benchmark: sequential reference stack vs batched fast stack vs MPS.

Engineering measurement, not science: times the goal-selection stage
(the wall-clock wall of every probe) three ways on one world --

  seq   reference FamilyVerifier, one rollout per candidate (the
        protocol every probe has used so far)
  fast  FastFamilyVerifier + score_goals: all candidates in one batch
  mps   the same, on Apple Metal if available

and reports the ranking overlap between seq and fast winners (RNG
streams differ, so scores match in distribution, not bitwise).
"""

from __future__ import annotations

import itertools
import time

import torch

from experiments.games_amodal.fast_family import FastFamilyVerifier
from experiments.games_amodal.fast_stack import (
    ENC_CANDIDATES, SLOTS, VALUES, score_goals)
from experiments.games_amodal.game_family import FamilyConfig, FamilyVerifier

torch.set_num_threads(1)
EPISODES, STEPS, SEED = 48, 12, 977


def slow_enc(prev, screen):
    return ENC_CANDIDATES["second2"](prev, screen)


def goal_cost(state, reference, goal):
    total = None
    for (a0, a1), (b0, b1), sign in goal:
        reach = ((state[:, a0] - reference[:, b0]).abs()
                 + (state[:, a1] - reference[:, b1]).abs()).float()
        term = sign * reach
        total = term if total is None else total + term
    return total


def slow_play(config, bank, goal, executor):
    v = FamilyVerifier(config, batch_size=EPISODES, seed=SEED)
    v.reset(seed=SEED)
    total = torch.zeros(EPISODES)
    prev = v.observation()
    for _ in range(STEPS):
        obs = v.observation()
        reference = slow_enc(prev, obs)
        reference = torch.where(reference < VALUES, reference,
                                torch.zeros_like(reference))
        best, action = None, torch.zeros(EPISODES, dtype=torch.long)
        for act in range(4):
            program = bank.get(act)
            state = (reference if program is None
                     else executor(program, reference))
            cost = goal_cost(state, reference, goal)
            if best is None:
                best = cost.clone()
            else:
                take = cost < best
                best = torch.where(take, cost, best)
                action = torch.where(take, torch.full((EPISODES,), act),
                                     action)
        total += v.step(action).reward
        prev = obs
    return float(total.mean())


_MOVES = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])


def move_executor(program, state):
    """Stand-in for the plant: slots 0/1 shift by the action's delta.

    program IS the action index here; the executor cost is identical
    across timing arms, but planning becomes real, so candidate scores
    genuinely differ and the seq/fast ranking check means something."""
    out = state.clone()
    delta = _MOVES[program].to(state.device)
    out[:, 0] = (state[:, 0] + delta[0]).clamp(0, 7)
    out[:, 1] = (state[:, 1] + delta[1]).clamp(0, 7)
    return out


def main():
    config = FamilyConfig(collect=1, avoid=1)
    bank = {a: a for a in range(4)}
    pairs = list(itertools.permutations(range(6), 2))
    goals = []
    for pa in pairs:
        for pb in pairs:
            if set(pa) & set(pb) or pa[0] > pa[1]:
                continue
            for sign in (1, -1):
                goals.append(((pa, pb, sign),))
    print(f"{len(goals)} candidate goals, {EPISODES} episodes x "
          f"{STEPS} steps each")

    t0 = time.perf_counter()
    seq_scores = torch.tensor(
        [slow_play(config, bank, g, move_executor) for g in goals])
    t_seq = time.perf_counter() - t0
    print(f"seq  (reference verifier): {t_seq:8.2f}s")

    def factory_cpu(batch):
        return FastFamilyVerifier(config, batch_size=batch, seed=SEED)

    t0 = time.perf_counter()
    fast_scores = score_goals(factory_cpu, goals, bank,
                              move_executor,
                              ENC_CANDIDATES["second2"], EPISODES,
                              STEPS, SEED)
    t_fast = time.perf_counter() - t0
    print(f"fast (batched, cpu):       {t_fast:8.2f}s   "
          f"speedup x{t_seq / t_fast:.1f}")

    top_seq = set(seq_scores.argsort(descending=True)[:8].tolist())
    top_fast = set(fast_scores.argsort(descending=True)[:8].tolist())
    print(f"top-8 overlap seq/fast: {len(top_seq & top_fast)}/8")

    if torch.backends.mps.is_available():
        def factory_mps(batch):
            return FastFamilyVerifier(config, batch_size=batch,
                                      seed=SEED, device="mps")
        t0 = time.perf_counter()
        score_goals(factory_mps, goals, bank, move_executor,
                    ENC_CANDIDATES["second2"], EPISODES, STEPS, SEED,
                    device="mps")
        t_mps = time.perf_counter() - t0
        print(f"mps  (batched, metal):     {t_mps:8.2f}s   "
              f"vs cpu x{t_fast / t_mps:.2f}")
    else:
        print("mps not available")


if __name__ == "__main__":
    main()
