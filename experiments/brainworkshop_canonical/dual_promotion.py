"""Holdout-population promotion for Dual acquisition.

Neural Workshop Dual 1-back from a blank file, plus one-step composition
onto Dual 2-back, is two-seed probation. This module freezes that protocol
and the promotion gate before any unused seed is run. The one-use holdout
is claimed only when ``--claim-holdout`` is set.

The warm Dual 2-back arm is composed execution, not a cheaper 2-back
search. This gate does not score a transfer ratio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_computer.promotion import (
    HoldoutLedger,
    MetricRequirement,
    PromotionEvidence,
    PromotionGate,
    evaluate_promotion,
    read_promotion_record,
    sha256_file,
    write_promotion_record,
)

from .controller_pretraining import load_temporal_controller_artifact
from .founding_promotion import combine_control_flags
from .neural_workshop_dual_acquisition_pilot import (
    NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA,
    run_neural_workshop_dual_acquisition,
)

EXPERIMENT_ID = "brainworkshop-dual-holdout-2026-08-15"
HOLDOUT_ID = "brainworkshop-dual-holdout-2026-08-15"
HOLDOUT_ATTEMPT_ID = "attempt-1"
DEVELOPMENT_POPULATION = "neural-workshop-dual-acq-dev-99117-99217"
PROMOTION_POPULATION = "neural-workshop-dual-holdout-2026-08-15"
PROTOCOL = "blank-file-dual-1back-then-compose-2back"
CONTROLLER_SHA256 = (
    "93a4dbb72953c60b7964546490c202ba9c6f4a4f9b4289de2c3f7db986690537"
)
DEVELOPMENT_REPORT_SHA256 = {
    99_117: "e3be7ccdabf02ba86e80d39227ff0c78c597bb923cd9edc4af28a616200f8f31",
    99_217: "848ba92724e4f88433424b82c546a39104ad6e0f180f72aed1be865b9d1580de",
}
DEVELOPMENT_SEEDS = (99_117, 99_217)
# Unused in Dual live, Dual acquisition, founding development, and the
# consumed founding holdout lease 110017-112017.
HOLDOUT_SEEDS = (113_017, 114_017, 115_017)
KNOWN_USED_SEEDS = frozenset(
    {
        17,
        18,
        *range(101, 117),
        81_017,
        91_017,
        93_017,
        94_017,
        95_017,
        96_017,
        97_017,
        98_017,
        98_117,
        99_017,
        99_117,
        99_217,
        110_017,
        111_017,
        112_017,
    }
)
REQUIRED_CONTROLS = (
    "fresh",
    "wrong_depth",
    "missing_history",
    "action_reversed",
    "controller_frozen",
)
THRESHOLD = 0.8
MINIMUM_BITS = 8
TRIALS = 60
SESSIONS = 6
MAXIMUM_DUAL_1BACK_BITS = 200.0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dual_acquisition_gate() -> PromotionGate:
    return PromotionGate(
        experiment_id=EXPERIMENT_ID,
        capability="dual-acquisition",
        development_population=DEVELOPMENT_POPULATION,
        promotion_population=PROMOTION_POPULATION,
        metric_requirements=(
            MetricRequirement("dual_1back_reached", minimum=1.0),
            MetricRequirement("dual_1back_bits", maximum=MAXIMUM_DUAL_1BACK_BITS),
            MetricRequirement("dual_1back_accuracy", minimum=THRESHOLD),
            MetricRequirement("dual_1back_retention", minimum=THRESHOLD),
            MetricRequirement("warm_dual_2back", minimum=THRESHOLD),
            MetricRequirement("warm_target_learning_bits", maximum=0.0),
            MetricRequirement("optimizer_updates", maximum=0.0),
            MetricRequirement("replayed_examples", maximum=0.0),
            MetricRequirement("audio_aligned", minimum=1.0),
        ),
        required_controls=REQUIRED_CONTROLS,
        min_replicates=3,
        max_workarounds=0,
    ).validate()


def development_manifest() -> dict[str, object]:
    return {
        "population": DEVELOPMENT_POPULATION,
        "seeds": list(DEVELOPMENT_SEEDS),
        "protocol": PROTOCOL,
        "report_sha256": {
            str(seed): digest for seed, digest in DEVELOPMENT_REPORT_SHA256.items()
        },
        "controller_sha256": CONTROLLER_SHA256,
        "schema": NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA,
    }


def promotion_manifest(*, seeds: tuple[int, ...] = HOLDOUT_SEEDS) -> dict[str, object]:
    return {
        "population": PROMOTION_POPULATION,
        "seeds": list(seeds),
        "protocol": PROTOCOL,
        "controller_sha256": CONTROLLER_SHA256,
        "trials": TRIALS,
        "sessions": SESSIONS,
        "threshold": THRESHOLD,
        "minimum_bits": MINIMUM_BITS,
        "schema": NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA,
    }


def development_manifest_digest() -> str:
    return _sha256_bytes(_canonical_json(development_manifest()))


def promotion_manifest_digest(*, seeds: tuple[int, ...] = HOLDOUT_SEEDS) -> str:
    return _sha256_bytes(_canonical_json(promotion_manifest(seeds=seeds)))


def configuration_digest(
    *,
    controller_sha256: str,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "trials": TRIALS,
                "sessions": SESSIONS,
                "threshold": THRESHOLD,
                "minimum_bits": MINIMUM_BITS,
                "holdout_seeds": list(seeds),
                "controller_sha256": controller_sha256,
                "dual_acquisition_gate": dual_acquisition_gate().digest(),
            }
        )
    )


def assert_unused_holdout_seeds(seeds: tuple[int, ...]) -> None:
    overlap = set(seeds) & KNOWN_USED_SEEDS
    if overlap:
        raise ValueError(
            f"holdout seeds collide with used Dual/founding seeds: {sorted(overlap)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("holdout seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("holdout population needs at least three seeds")


def require_controller(path: Path) -> str:
    digest = sha256_file(path)
    if digest != CONTROLLER_SHA256:
        raise ValueError(
            "controller digest does not match the frozen Dual acquisition controller"
        )
    return digest


def _packed(summary: Mapping[str, Any]) -> float:
    packed = summary.get("packed_exact_accuracy")
    if packed is None:
        raise ValueError("Dual report is missing packed_exact_accuracy")
    return float(packed)


def _audio_aligned(*summaries: Mapping[str, Any]) -> float:
    for summary in summaries:
        audio = int(summary["audio_events"])
        vision = int(summary["vision_events"])
        if audio <= 0 or audio != vision:
            return 0.0
    return 1.0


def extract_replicate_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Pull Dual acquisition gate metrics from one Neural Workshop report."""

    if report.get("schema") != NEURAL_WORKSHOP_DUAL_ACQUISITION_SCHEMA:
        raise ValueError("report is not a Neural Workshop Dual acquisition record")
    bits = report["dual_1back_bits_to_threshold"]
    train = report["dual_1back_train"]
    if not train:
        raise ValueError("Dual acquisition report has no Dual 1-back sessions")
    retention = report["dual_1back_retention"]
    warm = report["warm_dual_2back"]
    return {
        "dual_1back_reached": 1.0 if bits is not None else 0.0,
        "dual_1back_bits": float(bits) if bits is not None else float("inf"),
        "dual_1back_accuracy": _packed(train[-1]),
        "dual_1back_retention": _packed(retention),
        "warm_dual_2back": _packed(warm),
        "warm_target_learning_bits": float(report["warm_target_learning_bits"]),
        "optimizer_updates": float(report["optimizer_updates"]),
        "replayed_examples": float(report["replayed_examples"]),
        "audio_aligned": _audio_aligned(retention, warm),
        "program_file_updates": float(
            sum(int(row["program_file_updates"]) for row in train)
        ),
        "controller_frozen": (
            1.0 if all(bool(row.get("controller_frozen", True)) for row in train) else 0.0
        ),
    }


def _packed_below(summary: Mapping[str, Any], *, threshold: float) -> bool:
    packed = summary.get("packed_exact_accuracy")
    return packed is not None and float(packed) < threshold


def dual_control_flags(
    report: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    threshold: float = THRESHOLD,
) -> dict[str, bool]:
    selected = controls if controls is not None else report["controls"]
    train = report["dual_1back_train"]
    return {
        "fresh": (
            report["dual_1back_bits_to_threshold"] is not None
            and any(int(row["program_file_updates"]) > 0 for row in train)
        ),
        "wrong_depth": _packed_below(selected["wrong_depth"], threshold=threshold),
        "missing_history": _packed_below(
            selected["missing_history"], threshold=threshold
        ),
        "action_reversed": _packed_below(
            selected["action_reversed"], threshold=threshold
        ),
        "controller_frozen": (
            int(report["optimizer_updates"]) == 0
            and int(report["replayed_examples"]) == 0
        ),
    }


def current_git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()


def build_promotion_evidence(
    gate: PromotionGate,
    *,
    replicate_metrics: tuple[Mapping[str, float], ...],
    controls: Mapping[str, bool],
    artifact_hashes: Mapping[str, str],
    git_commit: str,
    controller_sha256: str,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
) -> PromotionEvidence:
    return PromotionEvidence(
        gate_digest=gate.digest(),
        holdout_id=HOLDOUT_ID,
        holdout_attempt_id=HOLDOUT_ATTEMPT_ID,
        development_manifest_digest=development_manifest_digest(),
        promotion_manifest_digest=promotion_manifest_digest(seeds=seeds),
        git_commit=git_commit,
        configuration_digest=configuration_digest(
            controller_sha256=controller_sha256, seeds=seeds
        ),
        artifact_hashes=dict(artifact_hashes),
        replicate_metrics=tuple(dict(row) for row in replicate_metrics),
        controls=dict(controls),
        search_attempts=1,
        workaround_count=0,
        holdout_uses=1,
    ).validate()


def run_holdout_campaign(
    controller_payload: dict[str, object],
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    controller_artifact: Path,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
    claim_holdout: bool = False,
    git_commit: str | None = None,
    trials: int = TRIALS,
    sessions: int = SESSIONS,
) -> dict[str, Any]:
    """Run the unused Dual holdout population against the frozen gate."""

    if trials != TRIALS or sessions != SESSIONS:
        raise ValueError("Dual holdout must use the frozen 60-trial / 6-session protocol")
    assert_unused_holdout_seeds(seeds)
    controller_digest = require_controller(controller_artifact)
    output_directory.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[2]
    commit = git_commit or current_git_commit(repository)
    started = time.perf_counter()
    seed_rows: list[dict[str, Any]] = []
    replicate_metrics: list[dict[str, float]] = []
    flag_rows: list[dict[str, bool]] = []
    artifact_hashes = {
        "controller": controller_digest,
        "campaign_runner": sha256_file(Path(__file__)),
        "acquisition_runner": sha256_file(
            Path(__file__).with_name("neural_workshop_dual_acquisition_pilot.py")
        ),
    }

    for seed in seeds:
        seed_dir = output_directory / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        report = run_neural_workshop_dual_acquisition(
            controller_payload,
            neural_workshop_directory,
            trials=trials,
            sessions=sessions,
            seed=seed,
        )
        report_path = seed_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        metrics = extract_replicate_metrics(report)
        flags = dual_control_flags(report)
        replicate_metrics.append(metrics)
        flag_rows.append(flags)
        seed_report = {
            "seed": seed,
            "metrics": metrics,
            "control_flags": flags,
            "controls": report["controls"],
            "dual_1back_bits_to_threshold": report["dual_1back_bits_to_threshold"],
            "wall_seconds": report["wall_seconds"],
        }
        (seed_dir / "seed_summary.json").write_text(
            json.dumps(seed_report, indent=2, sort_keys=True) + "\n"
        )
        artifact_hashes[f"seed{seed}_report"] = sha256_file(report_path)
        seed_rows.append(seed_report)

    controls = combine_control_flags(tuple(flag_rows))
    gate = dual_acquisition_gate()
    evidence = build_promotion_evidence(
        gate,
        replicate_metrics=tuple(replicate_metrics),
        controls=controls,
        artifact_hashes=artifact_hashes,
        git_commit=commit,
        controller_sha256=controller_digest,
        seeds=seeds,
    )
    ledger_path = output_directory / "holdout-ledger.jsonl"
    decision = evaluate_promotion(gate, evidence)
    claimed = False
    if claim_holdout:
        ledger = HoldoutLedger(ledger_path)
        ledger.claim(
            HOLDOUT_ID,
            promotion_manifest_digest(seeds=seeds),
            HOLDOUT_ATTEMPT_ID,
        )
        claimed = True
        decision = evaluate_promotion(gate, evidence, holdout_ledger=ledger)
        write_promotion_record(
            output_directory / "promotion_dual_acquisition.json",
            gate,
            evidence,
            decision,
            holdout_ledger=ledger,
        )
    else:
        write_promotion_record(
            output_directory / "promotion_dual_acquisition.json",
            gate,
            evidence,
            decision,
        )

    campaign = {
        "schema": "neural-computer.dual-holdout-campaign.v1",
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL,
        "holdout_id": HOLDOUT_ID,
        "holdout_claimed": claimed,
        "seeds": list(seeds),
        "controller_sha256": controller_digest,
        "git_commit": commit,
        "development_manifest_digest": development_manifest_digest(),
        "promotion_manifest_digest": promotion_manifest_digest(seeds=seeds),
        "configuration_digest": configuration_digest(
            controller_sha256=controller_digest, seeds=seeds
        ),
        "controls": controls,
        "replicates": seed_rows,
        "dual_acquisition": decision.canonical(),
        "wall_seconds": time.perf_counter() - started,
    }
    (output_directory / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    return campaign


def claim_existing_campaign(output_directory: Path) -> dict[str, Any]:
    """Consume the Dual lease against already-written unused-seed reports."""

    record_path = output_directory / "promotion_dual_acquisition.json"
    campaign_path = output_directory / "campaign.json"
    gate, evidence, recorded = read_promotion_record(record_path)
    if recorded.eligible:
        raise ValueError("Dual promotion record is already eligible")
    if evidence.holdout_id != HOLDOUT_ID:
        raise ValueError("promotion record targets a different Dual holdout")
    ledger = HoldoutLedger(output_directory / "holdout-ledger.jsonl")
    ledger.claim(
        evidence.holdout_id,
        evidence.promotion_manifest_digest,
        evidence.holdout_attempt_id,
    )
    decision = evaluate_promotion(gate, evidence, holdout_ledger=ledger)
    write_promotion_record(
        record_path,
        gate,
        evidence,
        decision,
        holdout_ledger=ledger,
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["holdout_claimed"] = True
    campaign["dual_acquisition"] = decision.canonical()
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    if not decision.eligible:
        raise RuntimeError(
            "Dual holdout claim did not make the record eligible: "
            + "; ".join(decision.reasons)
        )
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--neural-workshop", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "session_records" / "brainworkshop_dual_holdout_2026-08-15",
    )
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--claim-holdout",
        action="store_true",
        help="consume the one-use Dual holdout lease after the unused seeds finish",
    )
    parser.add_argument(
        "--claim-existing",
        action="store_true",
        help="claim the Dual lease from unused-seed reports already on disk",
    )
    arguments = parser.parse_args()
    if arguments.claim_existing:
        campaign = claim_existing_campaign(arguments.output_dir)
    else:
        if arguments.neural_workshop is None:
            raise ValueError("Dual holdout needs --neural-workshop")
        campaign = run_holdout_campaign(
            load_temporal_controller_artifact(arguments.controller_artifact),
            arguments.neural_workshop,
            arguments.output_dir,
            controller_artifact=arguments.controller_artifact,
            claim_holdout=arguments.claim_holdout,
        )
    print(json.dumps(campaign, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
