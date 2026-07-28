#!/usr/bin/env python3
"""Size a sweep to fit the wall-clock cap before running it.

Waiting half an hour for one number is the failure this exists to prevent. A
campaign should be built to land inside the cap, not truncated by it, and the
lever is almost always the budget grid: cost is roughly linear in total training
steps, and the largest budgets are both the most expensive and the least
informative, because that is where the arms have already converged.

    plan_campaign.py --arms 3 --seeds 12 --grid 96,160,256,384,512,768 \
        --seconds-per-1k-steps 11 --workers 6

Prints the projected wall clock and, when it does not fit, the largest prefix of
the grid that does.
"""
from __future__ import annotations

import argparse


def project(grid: list[int], arms: int, seeds: int, rate: float,
            workers: int) -> float:
    """Wall-clock seconds for the whole sweep at a measured per-step rate."""
    steps = sum(grid) * arms * seeds
    return steps / 1000.0 * rate / max(1, workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument(
        "--seconds-per-1k-steps", type=float, required=True,
        help="measure it with one pilot run; it is the only empirical input")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cap-seconds", type=float, default=300.0)
    args = parser.parse_args()

    grid = [int(v) for v in args.grid.split(",") if v]
    rate, cap = args.seconds_per_1k_steps, args.cap_seconds
    total = project(grid, args.arms, args.seeds, rate, args.workers)
    runs = len(grid) * args.arms * args.seeds
    print(f"grid {grid}")
    print(f"{runs} runs, {sum(grid) * args.arms * args.seeds:,} training steps")
    print(f"projected wall clock {total:.0f}s against a {cap:.0f}s cap "
          f"at {args.workers} workers")

    if total <= cap:
        print("FITS")
        return

    print("DOES NOT FIT. Options, cheapest first:")
    # dropping the top budgets is the cheapest real fix: they cost the most and
    # sit where both arms have already converged
    for keep in range(len(grid) - 1, 0, -1):
        shorter = grid[:keep]
        if project(shorter, args.arms, args.seeds, rate, args.workers) <= cap:
            print(f"  grid {shorter}  (drops the top {len(grid) - keep}, "
                  f"which is where the curves saturate)")
            break
    for seeds in range(args.seeds - 1, 3, -1):
        if project(grid, args.arms, seeds, rate, args.workers) <= cap:
            print(f"  seeds {seeds} instead of {args.seeds} "
                  f"(costs statistical power; prefer cutting the grid)")
            break
    need = total / cap
    print(f"  or split into {need:.1f} campaigns, each scored as it lands")


if __name__ == "__main__":
    main()
