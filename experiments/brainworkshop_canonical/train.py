"""Run a short canonical Brain Workshop composition smoke test.

This command intentionally performs no optimizer update and makes no learning
claim.  Its purpose is to verify that rendered symbol events, opaque keypress
feedback, one controller/memory, the keypress decoder, episodic context, and
the retention ledger operate in one end-to-end runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import NBackVerifier
from .runner import CanonicalBrainWorkshopAgent
from .trainer import audit_retention, evaluate_policy, train_reward_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-back", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--symbol-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--reader", choices=("context", "relation"), default="context")
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = CanonicalBrainWorkshopAgent(
        symbol_count=args.symbol_count,
        n_back=args.n_back,
        seed=args.seed,
        reader_kind=args.reader,
    )
    baseline_controls: dict[str, float] = {}
    post_controls: dict[str, float] = {}
    retention_audit: list[float] = []
    if args.updates == 0:
        verifier = NBackVerifier(
            batch_size=args.batch_size,
            n_back=args.n_back,
            steps=args.steps,
            symbol_count=args.symbol_count,
            seed=args.seed,
        )
        rollout = agent.rollout(verifier)
        history = []
    else:
        evaluation_seeds = tuple(args.seed + 1000 + index for index in range(3))
        baseline_controls = {
            "fresh": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
            ),
            "time_shuffle": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
                time_shuffle=True,
            ),
            "history_reset": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
                reset_history=True,
            ),
        }
        history = train_reward_only(
            agent,
            n_back=args.n_back,
            updates=args.updates,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        verifier = NBackVerifier(
            batch_size=args.batch_size,
            n_back=args.n_back,
            steps=args.steps,
            symbol_count=args.symbol_count,
            seed=args.seed + args.updates,
        )
        rollout = agent.rollout(verifier, sample=False, record_retention=True)
        post_controls = {
            "fresh": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
            ),
            "time_shuffle": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
                time_shuffle=True,
            ),
            "history_reset": evaluate_policy(
                agent,
                n_back=args.n_back,
                batch_size=args.batch_size,
                seeds=evaluation_seeds,
                steps=args.steps,
                reset_history=True,
            ),
        }
        retention_audit = audit_retention(
            agent,
            n_back=args.n_back,
            batch_size=args.batch_size,
            seeds=tuple(args.seed + 2000 + index for index in range(3)),
            steps=args.steps,
        )
    report = {
        "schema": "neural-computer.brainworkshop-canonical-smoke.v1",
        "status": "composition_smoke_only" if args.updates == 0 else "reward_only_pilot",
        "n_back": args.n_back,
        "reader": args.reader,
        "batch_size": args.batch_size,
        "steps": verifier.steps,
        "eligible_trials": verifier.eligible_trials,
        "eligible_accuracy": [float(value) for value in rollout.eligible_accuracy],
        "mean_eligible_accuracy": float(rollout.eligible_accuracy.mean()),
        "baseline_controls": baseline_controls,
        "post_controls": post_controls,
        "retention_audit": retention_audit,
        "unique_verifier_bits": (
            sum(row.unique_verifier_bits for row in history)
            if history
            else args.batch_size * verifier.eligible_trials
        ),
        "unique_logical_lifetimes": (
            sum(args.batch_size for _ in history) if history else args.batch_size
        ),
        "evaluation_verifier_bits_per_control": (
            args.batch_size * verifier.eligible_trials
        ),
        "optimizer_updates": args.updates,
        "replayed_examples": sum(row.replayed_examples for row in history),
        "training_history": [row.__dict__ for row in history],
        "keypress_decoder_schema": agent.keypress_decoder.configuration()["schema"],
        "retention_records": len(
            agent.retention.payload()["records"]
            if hasattr(agent.retention, "payload")
            else []
        ),
        "retention_protected": int(
            agent.retention.status(agent.capability_address).protected
        ),
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
