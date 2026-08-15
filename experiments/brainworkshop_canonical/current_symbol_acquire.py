"""Unused-seed current-symbol acquire for the prototype-match operator.

This does not admit a file to AgentBrain. Delay-address files stay in the
bank. The holdout population is disjoint from Dual/founding/development
seeds used for this operator.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file
from neural_computer.temporal_program import PROTOTYPE_MATCH_EXECUTION_SCHEMA

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .dual_promotion import CONTROLLER_SHA256, KNOWN_USED_SEEDS
from .lease_discrimination import (
    assert_discriminating,
    control_below_threshold_report,
    discrimination_report,
    separation_report,
)
from .program_search import search_temporal_programs
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import run_rendered_live_lifetime
from .seed_ledger import assert_unused_block, block

EXPERIMENT_ID = "brainworkshop-current-symbol-bound-frontend-2026-08-15"
DEVELOPMENT_SEEDS = (41,)
# 116017-118017 already ran the unbound random-encoder acquire.
PREVIOUS_ACQUIRE_SEEDS = frozenset({116_017, 117_017, 118_017})
HOLDOUT_SEEDS = (119_017, 120_017, 121_017)
BOUND_FRONTEND_SEEDS = frozenset(HOLDOUT_SEEDS)
SEARCH_LEASE_ID = "brainworkshop-current-symbol-search-lease-2026-08-15"
SEARCH_LEASE_SEEDS = (122_017, 123_017, 124_017)
# The 122017-124017 record predates `and` in the grammar and sits below the
# trial floor. This arm re-establishes the claim on a fresh block.
DISCRIMINATING_LEASE_ID = (
    "brainworkshop-current-symbol-lease-discriminating-2026-08-15"
)
DISCRIMINATING_BLOCK = "current_symbol_lease_discriminating"
DISCRIMINATING_STEPS = 448
LEASE_SESSIONS = 6
DIAGNOSTIC_SEEDS = frozenset({17, 41, 43, 201, 203})
FRONTEND_SEED = 1001
THRESHOLD = 0.8
MINIMUM_BITS = 24
STEPS = 48
LEARNING_RATE = 0.3


def current_symbol_config(*, steps: int = STEPS) -> RenderedBrainWorkshopConfig:
    return RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        match_rule="current_symbol",
        target_symbol=0,
    )


def assert_unused_holdout_seeds(seeds: tuple[int, ...]) -> None:
    used = (
        set(KNOWN_USED_SEEDS)
        | set(DEVELOPMENT_SEEDS)
        | DIAGNOSTIC_SEEDS
        | PREVIOUS_ACQUIRE_SEEDS
    )
    overlap = set(seeds) & used
    if overlap:
        raise ValueError(
            f"current-symbol holdout seeds collide with used seeds: {sorted(overlap)}"
        )
    if set(seeds) & {113_017, 114_017, 115_017}:
        raise ValueError("current-symbol holdout cannot reuse the Dual holdout lease")
    if set(seeds) & PREVIOUS_ACQUIRE_SEEDS:
        raise ValueError("current-symbol holdout cannot reuse the unbound acquire seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("holdout seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("holdout population needs at least three seeds")


def require_controller(path: Path) -> str:
    digest = sha256_file(path)
    if digest != CONTROLLER_SHA256:
        raise ValueError("controller digest does not match the frozen controller")
    return digest


def _machine(controller_payload: dict[str, object], *, learn: bool):
    machine = build_pretrained_controller_program_machine(
        controller_payload,
        learning_rate=LEARNING_RATE,
        sample=learn,
        inherit_program_prior=False,
    )
    machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
    machine.learning_enabled = learn
    machine.sample = learn
    return machine


def _encoders(machine) -> RenderedBrainWorkshopEncoders:
    encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)
    return encoders


def curated_frontend(
    machine, *, seed: int = FRONTEND_SEED, path: Path | None = None
) -> RenderedBrainWorkshopEncoders:
    if path is not None:
        encoders = RenderedBrainWorkshopEncoders.load(path)
        if (
            encoders.event_width != machine.event_width
            or encoders.source_key_width != machine.source_key_width
        ):
            raise ValueError("curated frontend geometry does not match the controller")
        return encoders
    return RenderedBrainWorkshopEncoders.seeded(
        machine.event_width,
        source_key_width=machine.source_key_width,
        seed=seed,
    )


def _summary(name: str, report, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "arm": name,
        "accuracy": report.eligible_accuracy,
        "unique_verifier_bits": report.unique_verifier_bits,
        "program_file_updates": report.program_file_updates,
        "optimizer_updates": report.optimizer_updates,
        "replayed_examples": report.replayed_examples,
    }
    if extra:
        row.update(extra)
    return row


def run_replicate(
    controller_payload: dict[str, object],
    *,
    seed: int,
    steps: int = STEPS,
    bank_path: Path | None = None,
    encoders: RenderedBrainWorkshopEncoders | None = None,
) -> dict[str, Any]:
    """Acquire one prototype file and score frozen hold plus reject controls."""

    machine = _machine(controller_payload, learn=True)
    if encoders is None:
        encoders = curated_frontend(machine)
    digest = machine.controller_digest()
    config = current_symbol_config(steps=steps)
    started = time.perf_counter()
    train = run_rendered_live_lifetime(
        machine, encoders, config, seed=seed, learn=True, sample=True
    )
    machine.learning_enabled = False
    machine.sample = False
    hold_seed = seed + 1
    hold = run_rendered_live_lifetime(
        machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    learned = machine.prototype.detach().clone()
    machine.prototype.data.zero_()
    zeros = run_rendered_live_lifetime(
        machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    machine.prototype.data.copy_(learned)
    reversed_actions = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        action_permutation=(1, 0),
    )
    shuffled = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        randomized_outcome_seed=seed,
    )
    other = _encoders(machine)
    crossed = run_rendered_live_lifetime(
        machine, other, config, seed=hold_seed, learn=False, sample=False
    )
    delay = None
    if bank_path is not None:
        bank = ExternalTemporalProgramBank.load_bank(bank_path)
        delay_machine = _machine(controller_payload, learn=False)
        delay_machine.load_admitted_program_artifact(
            bank.artifact(0), controller_digest=bank.controller_digest
        )
        delay_report = run_rendered_live_lifetime(
            delay_machine,
            encoders,
            config,
            seed=hold_seed,
            learn=False,
            sample=False,
        )
        delay = _summary("delay_slot0", delay_report)
    elapsed = time.perf_counter() - started
    controls = {
        "zeros": zeros.eligible_accuracy,
        "action_reversed": reversed_actions.eligible_accuracy,
        "reward_shuffled": shuffled.eligible_accuracy,
        "cross_encoder": crossed.eligible_accuracy,
    }
    if delay is not None:
        controls["delay_slot0"] = float(delay["accuracy"])
    label = max(controls, key=lambda name: controls[name])
    separation = separation_report(
        hold.eligible_accuracy, controls[label], steps, control_label=label
    )
    below_threshold = {
        name: control_below_threshold_report(
            accuracy, steps, threshold=THRESHOLD, control_label=name
        )
        for name, accuracy in controls.items()
    }
    hold_ok = hold.eligible_accuracy >= THRESHOLD
    zeros_ok = zeros.eligible_accuracy < THRESHOLD
    reverse_ok = reversed_actions.eligible_accuracy < THRESHOLD
    shuffle_ok = shuffled.eligible_accuracy < THRESHOLD
    delay_ok = delay is None or float(delay["accuracy"]) < THRESHOLD
    cross_ok = crossed.eligible_accuracy < THRESHOLD
    return {
        "schema": "neural-computer.current-symbol-acquire-replicate.v1",
        "separation": separation,
        "control_below_threshold": below_threshold,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "hold_seed": hold_seed,
        "steps": steps,
        "controller_digest": digest,
        "controller_unchanged": machine.controller_digest() == digest,
        "frontend_digest": encoders.digest(),
        "cross_frontend_digest": other.digest(),
        "prototype_norm": float(machine.prototype.detach().norm()),
        "train": _summary("train", train),
        "hold": _summary("hold", hold),
        "zeros": _summary("zeros", zeros),
        "action_reversed": _summary("action_reversed", reversed_actions),
        "reward_shuffled": _summary("reward_shuffled", shuffled),
        "cross_encoder": _summary("cross_encoder", crossed),
        "delay_slot0": delay,
        "elapsed_seconds": elapsed,
        "accepted": bool(
            hold_ok
            and bool(separation["separated"])
            and zeros_ok
            and reverse_ok
            and shuffle_ok
            and delay_ok
            and cross_ok
            and machine.controller_digest() == digest
            and encoders.digest() != other.digest()
            and hold.unique_verifier_bits >= MINIMUM_BITS
        ),
    }


def run_campaign(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
    steps: int = STEPS,
    frontend_path: Path | None = None,
    frontend_seed: int = FRONTEND_SEED,
    enforce_discrimination: bool = True,
) -> dict[str, Any]:
    """Run the unused current-symbol population. Never writes the bank."""

    assert_unused_holdout_seeds(seeds)
    # current-symbol scores every step, so eligible trials equal steps.
    discrimination = (
        assert_discriminating(steps, threshold=THRESHOLD)
        if enforce_discrimination
        else discrimination_report(steps, threshold=THRESHOLD)
    )
    controller_sha = require_controller(controller_path)
    before = sha256_file(bank_path)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    slot0 = bank.artifact(0).digest()
    payload = load_temporal_controller_artifact(controller_path)
    probe = _machine(payload, learn=False)
    encoders = curated_frontend(probe, seed=frontend_seed, path=frontend_path)
    frontend_digest = encoders.digest()
    output_directory.mkdir(parents=True, exist_ok=True)
    replicates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        report = run_replicate(
            payload,
            seed=seed,
            steps=steps,
            bank_path=bank_path,
            encoders=encoders,
        )
        path = output_directory / f"seed{seed}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        replicates.append(report)
    after = sha256_file(bank_path)
    restored = ExternalTemporalProgramBank.load_bank(bank_path)
    if after != before or restored.artifact(0).digest() != slot0:
        raise RuntimeError("current-symbol campaign mutated AgentBrain.bank")
    accepted = all(row["accepted"] for row in replicates) and bool(
        discrimination["discriminating"]
    )
    campaign = {
        "schema": "neural-computer.current-symbol-acquire-campaign.v1",
        "discrimination": discrimination,
        "experiment_id": EXPERIMENT_ID,
        "status": "replicated_not_admitted" if accepted else "rejected",
        "controller_sha256": controller_sha,
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "slot0_digest": slot0,
        "frontend_digest": frontend_digest,
        "frontend_shared": all(
            row["frontend_digest"] == frontend_digest for row in replicates
        ),
        "admitted": False,
        "seeds": list(seeds),
        "steps": steps,
        "threshold": THRESHOLD,
        "minimum_bits": MINIMUM_BITS,
        "replicates": replicates,
        "accepted": accepted,
        "elapsed_seconds": time.perf_counter() - started,
        "unique_verifier_bits": {
            f"hold_seed{row['seed']}": int(row["hold"]["unique_verifier_bits"])
            for row in replicates
        },
        "program_file_updates": sum(
            int(row["train"]["program_file_updates"]) for row in replicates
        ),
        "optimizer_updates": 0,
        "replayed_examples": 0,
    }
    (output_directory / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    ledger = {
        "schema": "neural-computer.sample-efficiency-ledger.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": campaign["status"],
        "source": "in_repository_run",
        "seed_blocks": [[seed, seed] for seed in seeds],
        "replication_count": len(seeds),
        "unique_verifier_bits": campaign["unique_verifier_bits"],
        "unique_logical_lifetimes_all_arms": len(seeds) * 6,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": {
            f"seed{row['seed']}": row["elapsed_seconds"] for row in replicates
        },
        "stable_bits_to_threshold": {
            f"hold_seed{row['seed']}": int(row["hold"]["unique_verifier_bits"])
            if row["accepted"]
            else None
            for row in replicates
        },
        "retention_on_mastered_primitive": (
            "current-symbol frozen hold on unused seeds; zeros remain below threshold"
        ),
        "transfer_ratio_against_fresh_learner": None,
        "transfer_ratio_note": "zeros are the unmatched invent file, not a fresh delay learner",
        "controls": {
            "separated_from_best_control": all(
                bool(row["separation"]["separated"]) for row in replicates
            ),
            "worst_separation_probability": max(
                float(row["separation"]["control_reproduces_winner_probability"])
                for row in replicates
            ),
            "smallest_winner_control_margin": min(
                float(row["separation"]["margin"]) for row in replicates
            ),
            "discriminating_episode": bool(discrimination["discriminating"]),
            "near_miss_pass_probability": discrimination[
                "near_miss_pass_probability"
            ],
            "zeros_below_threshold": all(
                float(row["zeros"]["accuracy"]) < THRESHOLD for row in replicates
            ),
            "action_reversed_below_threshold": all(
                float(row["action_reversed"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "reward_shuffled_below_threshold": all(
                float(row["reward_shuffled"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "delay_slot0_below_threshold": all(
                row["delay_slot0"] is None
                or float(row["delay_slot0"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "cross_encoder_below_threshold": all(
                float(row["cross_encoder"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "controller_frozen": all(row["controller_unchanged"] for row in replicates),
            "bank_unchanged": after == before,
            "admitted": False,
        },
        "unresolved": {
            "bank_admission": "prototype-match is not a curated AgentBrain slot",
            "open_program_induction": "grammar is still retrieve, compose, invent",
            "desktop_dual": "not a trainer and not this experiment",
        },
    }
    (output_directory / "sample_efficiency_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )
    checksums = []
    for path in sorted(output_directory.glob("*.json")):
        checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_directory / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return campaign


def assert_unused_search_lease_seeds(seeds: tuple[int, ...]) -> None:
    used = (
        set(KNOWN_USED_SEEDS)
        | set(DEVELOPMENT_SEEDS)
        | DIAGNOSTIC_SEEDS
        | PREVIOUS_ACQUIRE_SEEDS
        | BOUND_FRONTEND_SEEDS
    )
    overlap = set(seeds) & used
    if overlap:
        raise ValueError(
            f"search-lease seeds collide with used seeds: {sorted(overlap)}"
        )
    if set(seeds) & {113_017, 114_017, 115_017}:
        raise ValueError("search lease cannot reuse the Dual holdout lease")
    if len(set(seeds)) != len(seeds):
        raise ValueError("search-lease seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("search lease needs at least three seeds")


def _stable_hold_bits(
    holds: list[dict[str, Any]],
    *,
    threshold: float = THRESHOLD,
    minimum_bits: int = MINIMUM_BITS,
) -> int | None:
    total = 0
    for index, row in enumerate(holds):
        bits = int(row["unique_verifier_bits"])
        total += bits
        if bits < minimum_bits:
            continue
        if all(
            float(item["accuracy"]) >= threshold
            and int(item["unique_verifier_bits"]) >= minimum_bits
            for item in holds[index:]
        ):
            return total
    return None


def run_search_lease_replicate(
    controller_payload: dict[str, object],
    bank_path: Path,
    encoders,
    *,
    seed: int,
    steps: int = STEPS,
    sessions: int = LEASE_SESSIONS,
) -> dict[str, Any]:
    """Search invents, acquires, binds the frontend, then measures a hold prefix."""

    from neural_computer import ExternalTemporalProgramBank

    machine = _machine(controller_payload, learn=False)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    config = current_symbol_config(steps=steps)
    started = time.perf_counter()

    def acquire(proposal):
        del proposal
        machine.learning_enabled = True
        machine.sample = True
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed, learn=True, sample=True
        )
        machine.learning_enabled = False
        machine.sample = False
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
            "program_file_updates": report.program_file_updates,
        }

    def evaluate(proposal):
        del proposal
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=seed + 1,
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    search = search_temporal_programs(
        bank,
        machine,
        evaluate,
        threshold=THRESHOLD,
        minimum_bits=MINIMUM_BITS,
        acquire=acquire,
        encoders=encoders,
    )
    holds = []
    if search["winner"] is not None:
        holds.append(
            {
                "session": 0,
                "seed": seed + 1,
                "accuracy": search["winner"]["accuracy"],
                "unique_verifier_bits": search["winner"]["unique_verifier_bits"],
            }
        )
        for index in range(1, sessions):
            report = run_rendered_live_lifetime(
                machine,
                encoders,
                config,
                seed=seed + 1 + index,
                learn=False,
                sample=False,
            )
            holds.append(
                {
                    "session": index,
                    "seed": seed + 1 + index,
                    "accuracy": report.eligible_accuracy,
                    "unique_verifier_bits": report.unique_verifier_bits,
                }
            )
    hold_seed = seed + sessions
    zeros_machine = _machine(controller_payload, learn=False)
    zeros = run_rendered_live_lifetime(
        zeros_machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    reversed_actions = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        action_permutation=(1, 0),
    )
    shuffled = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        randomized_outcome_seed=seed,
    )
    other = _encoders(machine)
    crossed = run_rendered_live_lifetime(
        machine, other, config, seed=hold_seed, learn=False, sample=False
    )
    delay_machine = _machine(controller_payload, learn=False)
    delay_machine.load_admitted_program_artifact(
        bank.artifact(0), controller_digest=bank.controller_digest
    )
    delay = run_rendered_live_lifetime(
        delay_machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    controls = {
        "zeros": zeros.eligible_accuracy,
        "action_reversed": reversed_actions.eligible_accuracy,
        "reward_shuffled": shuffled.eligible_accuracy,
        "cross_encoder": crossed.eligible_accuracy,
        "delay_slot0": delay.eligible_accuracy,
    }
    worst_hold = min((float(row["accuracy"]) for row in holds), default=0.0)
    label = max(controls, key=lambda name: controls[name])
    separation = separation_report(
        worst_hold, controls[label], steps, control_label=label
    )
    below_threshold = {
        name: control_below_threshold_report(
            accuracy, steps, threshold=THRESHOLD, control_label=name
        )
        for name, accuracy in controls.items()
    }
    stable = _stable_hold_bits(holds)
    winner = search["winner"]
    accepted = bool(
        winner is not None
        and winner["kind"] == "invent"
        and winner.get("frontend_digest") == encoders.digest()
        and stable is not None
        and zeros.eligible_accuracy < THRESHOLD
        and reversed_actions.eligible_accuracy < THRESHOLD
        and shuffled.eligible_accuracy < THRESHOLD
        and crossed.eligible_accuracy < THRESHOLD
        and delay.eligible_accuracy < THRESHOLD
        and bool(separation["separated"])
        and machine.controller_digest() == bank.controller_digest
    )
    return {
        "schema": "neural-computer.current-symbol-search-lease-replicate.v1",
        "separation": separation,
        "control_below_threshold": below_threshold,
        "experiment_id": SEARCH_LEASE_ID,
        "seed": seed,
        "steps": steps,
        "sessions": sessions,
        "search": search,
        "holds": holds,
        "stable_bits_to_threshold": stable,
        "frontend_digest": encoders.digest(),
        "controller_digest": machine.controller_digest(),
        "zeros": _summary("zeros", zeros),
        "action_reversed": _summary("action_reversed", reversed_actions),
        "reward_shuffled": _summary("reward_shuffled", shuffled),
        "cross_encoder": _summary("cross_encoder", crossed),
        "delay_slot0": _summary("delay_slot0", delay),
        "elapsed_seconds": time.perf_counter() - started,
        "accepted": accepted,
    }


def run_search_lease(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = SEARCH_LEASE_SEEDS,
    steps: int = STEPS,
    sessions: int = LEASE_SESSIONS,
    frontend_path: Path | None = None,
    frontend_seed: int = FRONTEND_SEED,
    block_name: str | None = None,
    enforce_discrimination: bool = True,
) -> dict[str, Any]:
    """Longer unused-seed search lease. Never writes the bank."""

    if block_name is None:
        assert_unused_search_lease_seeds(seeds)
    else:
        assert_unused_block(block_name, seeds, sessions=sessions)
    discrimination = (
        assert_discriminating(steps, threshold=THRESHOLD)
        if enforce_discrimination
        else discrimination_report(steps, threshold=THRESHOLD)
    )
    controller_sha = require_controller(controller_path)
    before = sha256_file(bank_path)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    slot0 = bank.artifact(0).digest()
    payload = load_temporal_controller_artifact(controller_path)
    probe = _machine(payload, learn=False)
    encoders = curated_frontend(probe, seed=frontend_seed, path=frontend_path)
    frontend_digest = encoders.digest()
    output_directory.mkdir(parents=True, exist_ok=True)
    replicates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        report = run_search_lease_replicate(
            payload,
            bank_path,
            encoders,
            seed=seed,
            steps=steps,
            sessions=sessions,
        )
        path = output_directory / f"seed{seed}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        replicates.append(report)
    after = sha256_file(bank_path)
    restored = ExternalTemporalProgramBank.load_bank(bank_path)
    if after != before or restored.artifact(0).digest() != slot0:
        raise RuntimeError("search lease mutated AgentBrain.bank")
    accepted = all(row["accepted"] for row in replicates) and bool(
        discrimination["discriminating"]
    )
    campaign = {
        "schema": "neural-computer.current-symbol-search-lease.v1",
        "discrimination": discrimination,
        "experiment_id": SEARCH_LEASE_ID,
        "status": "replicated_not_admitted" if accepted else "rejected",
        "controller_sha256": controller_sha,
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "slot0_digest": slot0,
        "frontend_digest": frontend_digest,
        "frontend_shared": all(
            row["frontend_digest"] == frontend_digest for row in replicates
        ),
        "admitted": False,
        "selection": "search_invent_bound_frontend",
        "seeds": list(seeds),
        "steps": steps,
        "sessions": sessions,
        "threshold": THRESHOLD,
        "minimum_bits": MINIMUM_BITS,
        "replicates": replicates,
        "accepted": accepted,
        "elapsed_seconds": time.perf_counter() - started,
        "unique_verifier_bits": {
            f"hold_seed{row['seed']}": sum(
                int(item["unique_verifier_bits"]) for item in row["holds"]
            )
            for row in replicates
        },
        "stable_bits_to_threshold": {
            f"seed{row['seed']}": row["stable_bits_to_threshold"]
            for row in replicates
        },
        "optimizer_updates": 0,
        "replayed_examples": 0,
    }
    (output_directory / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    ledger = {
        "schema": "neural-computer.sample-efficiency-ledger.v1",
        "experiment_id": SEARCH_LEASE_ID,
        "status": campaign["status"],
        "source": "in_repository_run",
        "seed_blocks": [[seed, seed] for seed in seeds],
        "replication_count": len(seeds),
        "unique_verifier_bits": campaign["unique_verifier_bits"],
        "unique_logical_lifetimes_all_arms": len(seeds) * (sessions + 5),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": {
            f"seed{row['seed']}": row["elapsed_seconds"] for row in replicates
        },
        "stable_bits_to_threshold": campaign["stable_bits_to_threshold"],
        "retention_on_mastered_primitive": (
            "search-invented prototype bound to curated frontend; "
            "stable hold prefix on unused seeds"
        ),
        "transfer_ratio_against_fresh_learner": None,
        "transfer_ratio_note": "zeros and delay slot 0 are reject controls, not a Dual climb",
        "controls": {
            "separated_from_best_control": all(
                bool(row["separation"]["separated"]) for row in replicates
            ),
            "worst_separation_probability": max(
                float(row["separation"]["control_reproduces_winner_probability"])
                for row in replicates
            ),
            "smallest_winner_control_margin": min(
                float(row["separation"]["margin"]) for row in replicates
            ),
            "discriminating_episode": bool(discrimination["discriminating"]),
            "near_miss_pass_probability": discrimination[
                "near_miss_pass_probability"
            ],
            "zeros_below_threshold": all(
                float(row["zeros"]["accuracy"]) < THRESHOLD for row in replicates
            ),
            "action_reversed_below_threshold": all(
                float(row["action_reversed"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "reward_shuffled_below_threshold": all(
                float(row["reward_shuffled"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "delay_slot0_below_threshold": all(
                float(row["delay_slot0"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "cross_encoder_below_threshold": all(
                float(row["cross_encoder"]["accuracy"]) < THRESHOLD
                for row in replicates
            ),
            "frontend_bound": all(
                (row["search"]["winner"] or {}).get("frontend_digest")
                == frontend_digest
                for row in replicates
            ),
            "bank_unchanged": after == before,
            "admitted": False,
        },
        "unresolved": {
            "bank_admission": "bound prototype is not a curated AgentBrain slot",
            "open_program_induction": "grammar is still retrieve, compose, invent",
            "dual_2back_transfer": "not this experiment",
        },
    }
    (output_directory / "sample_efficiency_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )
    checksums = []
    for path in sorted(output_directory.glob("*.json")):
        checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_directory / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_current_symbol_bound_frontend_2026-08-15"
        ),
    )
    parser.add_argument("--frontend", type=Path, default=None)
    parser.add_argument("--frontend-seed", type=int, default=FRONTEND_SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--search-lease",
        action="store_true",
        help="unused-seed search invent/acquire lease; does not write the bank",
    )
    parser.add_argument(
        "--discriminating-lease",
        action="store_true",
        help=(
            "search lease on a fresh block at the eligible-trial floor; this "
            "is the arm that stands under the current grammar"
        ),
    )
    parser.add_argument("--sessions", type=int, default=LEASE_SESSIONS)
    arguments = parser.parse_args()
    if arguments.discriminating_lease:
        repository_records = Path(__file__).parents[2] / "session_records"
        if arguments.output_dir == (
            Path(__file__).parents[2]
            / "session_records"
            / "brainworkshop_current_symbol_bound_frontend_2026-08-15"
        ):
            arguments.output_dir = (
                repository_records
                / "brainworkshop_current_symbol_lease_discriminating_2026-08-15"
            )
        campaign = run_search_lease(
            arguments.controller_artifact,
            arguments.bank,
            arguments.output_dir,
            seeds=block(DISCRIMINATING_BLOCK),
            steps=(
                DISCRIMINATING_STEPS
                if arguments.steps == STEPS
                else arguments.steps
            ),
            sessions=arguments.sessions,
            frontend_path=arguments.frontend,
            frontend_seed=arguments.frontend_seed,
            block_name=DISCRIMINATING_BLOCK,
        )
    elif arguments.search_lease:
        if arguments.output_dir == (
            Path(__file__).parents[2]
            / "session_records"
            / "brainworkshop_current_symbol_bound_frontend_2026-08-15"
        ):
            arguments.output_dir = (
                Path(__file__).parents[2]
                / "session_records"
                / "brainworkshop_current_symbol_search_lease_2026-08-15"
            )
        campaign = run_search_lease(
            arguments.controller_artifact,
            arguments.bank,
            arguments.output_dir,
            steps=arguments.steps,
            sessions=arguments.sessions,
            frontend_path=arguments.frontend,
            frontend_seed=arguments.frontend_seed,
            enforce_discrimination=False,
        )
    else:
        campaign = run_campaign(
            arguments.controller_artifact,
            arguments.bank,
            arguments.output_dir,
            steps=arguments.steps,
            frontend_path=arguments.frontend,
            frontend_seed=arguments.frontend_seed,
            enforce_discrimination=False,
        )
    print(json.dumps({
        "accepted": campaign["accepted"],
        "status": campaign["status"],
        "admitted": campaign["admitted"],
        "bank_unchanged": campaign["bank_unchanged"],
        "seeds": campaign["seeds"],
        "elapsed_seconds": campaign["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
