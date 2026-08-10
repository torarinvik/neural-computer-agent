"""Audit whether verifier admission accepts genuinely useful transfer.

This uses the same frozen-core, adaptive known curriculum, isolated challenger,
causal controls, and retention gates as ``novel_challenger``. The target and
observation mask are both unseen, but the target is a nearby continuation of
the mastered successor. A successful audit must select the copied prior rather
than merely proving that fresh state is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.policy_free_intention_routing import novel_challenger

POSITIVE_TRANSFER_TASK = novel_challenger.ChallengerTask(
    task_id="unseen_nearby_target_positive_transfer",
    context_mask=torch.tensor(
        [True, True, True, False, True, True, False, True, False, True, False, True]
    ),
    target=torch.tensor([0.45, -0.82]),
    expected_initialization="transfer",
    probe_direction="transfer",
    report_schema="neural-computer.policy-free-intention-positive-transfer-challenger.v1",
    claim_boundary=(
        "bounded verifier-selected positive transfer for one unseen nearby "
        "evidence combination and target after a known adaptive sequence; broad "
        "positive transfer, arbitrary new computation, and general continual "
        "learning remain unqualified"
    ),
).validate()


def run(seed: int, report_out: Path) -> dict[str, object]:
    return novel_challenger.run(
        seed,
        report_out,
        task=POSITIVE_TRANSFER_TASK,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
