"""Black-box audiovisual leak checks for the Python reference environment.

The test constructs pairs of private questions with identical public cards and
different hidden evaluator fields.  The model-visible packets must remain
identical at every point in the same action sequence.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .environment import Action, RealtimeEpisode, generate_question


class FrozenClock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance(self, milliseconds: int):
        self.ns += milliseconds * 1_000_000


def _packet_equal(left, right) -> bool:
    return (left.timestamp_ms == right.timestamp_ms and
            np.array_equal(left.frame, right.frame) and
            np.array_equal(left.pcm, right.pcm))


def check(seed: int) -> dict:
    original = generate_question(seed, premises=4)
    # Keep every rendered field unchanged, while changing hidden answer and
    # family metadata.  RealtimeEpisode never receives either field in a packet.
    alternate = replace(original, answer=not original.answer,
                        family="hidden-alternate")
    left_clock, right_clock = FrozenClock(), FrozenClock()
    left = RealtimeEpisode(original, deadline_ms=10_000, clock_ns=left_clock)
    right = RealtimeEpisode(alternate, deadline_ms=10_000, clock_ns=right_clock)
    actions = (Action.WAIT, Action.NEXT, Action.NEXT, Action.PREVIOUS,
               Action.NEXT, Action.NEXT, Action.NEXT, Action.NEXT)
    checks = []
    for action in actions:
        left_result, right_result = left.step(action), right.step(action)
        checks.append(_packet_equal(left_result.observation, right_result.observation))
        left_clock.advance(17)
        right_clock.advance(17)
    return {"seed": seed, "identical_packets": all(checks),
            "steps": len(checks), "failed_steps": [i for i, ok in enumerate(checks) if not ok]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("boundary_leak.json"))
    args = parser.parse_args()
    rows = [check(seed) for seed in range(args.count)]
    payload = {"schema": "syllogimous.boundary-leak.v1", "count": len(rows),
               "failed": [row for row in rows if not row["identical_packets"]],
               "passed": all(row["identical_packets"] for row in rows)}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"count": args.count, "failed": len(payload["failed"]),
                      "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
