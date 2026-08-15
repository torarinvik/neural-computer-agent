"""Holdout-population promotion for header transfer and first-time depth.

Development seeds 94017 and 97017 stay in the probation record. This module
freezes the skip-shallower protocol and the promotion gates before any unused
seed is run. A third development seed is not a promotion. The one-use holdout
is claimed only when ``--claim-holdout`` is set.
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

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import (
    HoldoutLedger,
    MetricRequirement,
    PromotionEvidence,
    PromotionGate,
    evaluate_promotion,
    sha256_file,
    write_promotion_record,
)

from .controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from .neural_workshop_autonomous_founding_pilot import (
    NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA,
    _live_config,
    run_autonomous_founding,
)
from .neural_workshop_live import (
    NeuralWorkshopIntervention,
    build_neural_workshop_environment,
    run_neural_workshop_live_lifetime,
)

EXPERIMENT_ID = "brainworkshop-founding-holdout-2026-08-15"
HOLDOUT_ID = "brainworkshop-founding-holdout-2026-08-15"
HOLDOUT_ATTEMPT_ID = "attempt-1"
DEVELOPMENT_POPULATION = "neural-workshop-founding-dev-94017-97017"
PROMOTION_POPULATION = "neural-workshop-founding-holdout-2026-08-15"
PROTOCOL = "skip-shallower-after-nearest-depth-miss"
SOURCE_BANK_SHA256 = (
    "fb43f74caafa314a31951c56e8412179d91721499c275af98f8d9824ea0ee633"
)
DEVELOPMENT_REPORT_SHA256 = {
    94_017: "301723844fa20862b076e73270a3fec2d2a5c30ca5a7550b619aac0cb7b1dbe4",
    97_017: "0c96493d47a6792049922d39f386cf52b50d9e60058c24bd193bcdec81bfbf42",
}
DEVELOPMENT_SEEDS = (94_017, 97_017)
# Unused in every prior Neural Workshop / Dual / founding / sealed-frontier run.
HOLDOUT_SEEDS = (110_017, 111_017, 112_017)
KNOWN_DEVELOPMENT_SEEDS = frozenset(
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
    }
)
RETRIEVE_KINDS = frozenset({"exact", "invariant", "try_existing"})
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def header_transfer_gate() -> PromotionGate:
    return PromotionGate(
        experiment_id=EXPERIMENT_ID,
        capability="header-transfer",
        development_population=DEVELOPMENT_POPULATION,
        promotion_population=PROMOTION_POPULATION,
        metric_requirements=(
            MetricRequirement("header_fresh_over_warm", minimum=2.0),
            MetricRequirement("header_warm_bits", maximum=50.0),
            MetricRequirement("header_accuracy", minimum=THRESHOLD),
            MetricRequirement("header_retrieved", minimum=1.0),
            MetricRequirement("retention_min", minimum=THRESHOLD),
            MetricRequirement("optimizer_updates", maximum=0.0),
            MetricRequirement("replayed_examples", maximum=0.0),
        ),
        required_controls=REQUIRED_CONTROLS,
        min_replicates=3,
        max_workarounds=0,
    ).validate()


def first_time_depth_gate() -> PromotionGate:
    return PromotionGate(
        experiment_id=EXPERIMENT_ID,
        capability="first-time-depth-invention",
        development_population=DEVELOPMENT_POPULATION,
        promotion_population=PROMOTION_POPULATION,
        metric_requirements=(
            MetricRequirement("depth_fresh_over_warm", minimum=1.25),
            MetricRequirement("depth_accuracy", minimum=THRESHOLD),
            MetricRequirement("depth_composed", minimum=1.0),
            MetricRequirement("retention_min", minimum=THRESHOLD),
            MetricRequirement("optimizer_updates", maximum=0.0),
            MetricRequirement("replayed_examples", maximum=0.0),
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
        "source_bank_sha256": SOURCE_BANK_SHA256,
        "schema": NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA,
    }


def promotion_manifest(*, seeds: tuple[int, ...] = HOLDOUT_SEEDS) -> dict[str, object]:
    return {
        "population": PROMOTION_POPULATION,
        "seeds": list(seeds),
        "protocol": PROTOCOL,
        "source_bank_sha256": SOURCE_BANK_SHA256,
        "trials": TRIALS,
        "threshold": THRESHOLD,
        "minimum_bits": MINIMUM_BITS,
        "schema": NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA,
    }


def development_manifest_digest() -> str:
    return _sha256_bytes(_canonical_json(development_manifest()))


def promotion_manifest_digest(*, seeds: tuple[int, ...] = HOLDOUT_SEEDS) -> str:
    return _sha256_bytes(_canonical_json(promotion_manifest(seeds=seeds)))


def configuration_digest(
    *,
    source_bank_sha256: str,
    controller_sha256: str,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "trials": TRIALS,
                "threshold": THRESHOLD,
                "minimum_bits": MINIMUM_BITS,
                "holdout_seeds": list(seeds),
                "source_bank_sha256": source_bank_sha256,
                "controller_sha256": controller_sha256,
                "header_transfer_gate": header_transfer_gate().digest(),
                "first_time_depth_gate": first_time_depth_gate().digest(),
            }
        )
    )


def _last_accuracy(arm: Mapping[str, Any]) -> float:
    attempts = arm["attempts"]
    if not attempts:
        raise ValueError("founding arm has no attempts")
    return float(attempts[-1]["accuracy"])


def _retention_min(report: Mapping[str, Any]) -> float:
    scores = [float(row["accuracy"]) for row in report["source_retention"]]
    if len(scores) != 3:
        raise ValueError("founding report must retain 1-back, 2-back, and 3-back")
    return min(scores)


def extract_replicate_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Pull gate metrics from one autonomous-founding report."""

    if report.get("schema") != NEURAL_WORKSHOP_AUTONOMOUS_FOUNDING_SCHEMA:
        raise ValueError("report is not an autonomous founding record")
    founding = report["founding"]
    header = report["warm_3cell_3back"]
    depth = report["warm_2cell_3back"]
    header_ratio = founding["header_fresh_over_warm"]
    depth_ratio = founding["depth_fresh_over_warm"]
    if header_ratio is None or depth_ratio is None:
        raise ValueError("founding report is missing a transfer ratio")
    return {
        "header_fresh_over_warm": float(header_ratio),
        "header_warm_bits": float(founding["header_warm_bits"]),
        "header_fresh_bits": float(founding["header_fresh_bits"]),
        "header_accuracy": _last_accuracy(header),
        "header_retrieved": (
            1.0 if header["resolved_kind"] in RETRIEVE_KINDS else 0.0
        ),
        "depth_fresh_over_warm": float(depth_ratio),
        "depth_warm_bits": float(founding["depth_warm_bits"]),
        "depth_fresh_bits": float(founding["depth_fresh_bits"]),
        "depth_accuracy": _last_accuracy(depth),
        "depth_composed": 1.0 if depth["resolved_kind"] == "compose" else 0.0,
        "retention_min": _retention_min(report),
        "optimizer_updates": float(report["optimizer_updates"]),
        "replayed_examples": float(report["replayed_examples"]),
        "program_file_updates": float(report["program_file_updates"]),
        "controller_frozen": 1.0 if report["controller_frozen"] else 0.0,
    }


def _below_threshold(summary: Mapping[str, Any], *, threshold: float) -> bool:
    accuracy = summary.get("accuracy")
    return accuracy is not None and float(accuracy) < threshold


def founding_control_flags(
    report: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = THRESHOLD,
) -> dict[str, bool]:
    fresh = report["fresh_3cell_3back"]
    founding = report["founding"]
    return {
        "fresh": (
            fresh["resolved_kind"] == "compose"
            and int(founding["header_fresh_bits"]) > int(founding["header_warm_bits"])
        ),
        "wrong_depth": _below_threshold(controls["wrong_depth"], threshold=threshold),
        "missing_history": _below_threshold(
            controls["missing_history"], threshold=threshold
        ),
        "action_reversed": _below_threshold(
            controls["action_reversed"], threshold=threshold
        ),
        "controller_frozen": bool(report["controller_frozen"])
        and int(report["optimizer_updates"]) == 0
        and int(report["replayed_examples"]) == 0,
    }


def combine_control_flags(
    per_seed: tuple[Mapping[str, bool], ...],
) -> dict[str, bool]:
    if not per_seed:
        raise ValueError("control flags require at least one seed")
    names = set(per_seed[0])
    if any(set(row) != names for row in per_seed):
        raise ValueError("control flag names must match across seeds")
    return {name: all(bool(row[name]) for row in per_seed) for name in sorted(names)}


def assert_unused_holdout_seeds(seeds: tuple[int, ...]) -> None:
    overlap = set(seeds) & KNOWN_DEVELOPMENT_SEEDS
    if overlap:
        raise ValueError(f"holdout seeds collide with development seeds: {sorted(overlap)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("holdout seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("holdout population needs at least three seeds")


def require_source_bank(path: Path) -> str:
    digest = sha256_file(path)
    if digest != SOURCE_BANK_SHA256:
        raise ValueError(
            "source instruction bank digest does not match the frozen founding bank"
        )
    return digest


def _summary(report: Any) -> dict[str, Any]:
    return {
        "accuracy": report.verifier_accuracy,
        "unique_verifier_bits": report.unique_verifier_bits,
        "program_file_updates": report.program_file_updates,
        "controller_frozen": report.controller_frozen,
    }


def execute_loaded_program(
    machine,
    bank: ExternalTemporalProgramBank,
    artifact,
    neural_workshop_directory: Path,
    *,
    n_back: int,
    active_cells: int,
    trials: int,
    seed: int,
    intervention: NeuralWorkshopIntervention | None = None,
) -> dict[str, Any]:
    config = _live_config(
        machine, n_back=n_back, active_cells=active_cells, trials=trials
    )
    environment, verifier = build_neural_workshop_environment(
        neural_workshop_directory, config, seed=seed
    )
    try:
        machine.load_recursive_program_artifact(
            artifact, controller_digest=bank.controller_digest
        )
        report = run_neural_workshop_live_lifetime(
            machine,
            config,
            seed=seed,
            environment=environment,
            verifier=verifier,
            learn=False,
            sample=False,
            intervention=intervention,
        )
    except Exception:
        environment.close()
        raise
    return _summary(report)


def run_founding_controls(
    controller_payload: dict[str, object],
    founding_report: Mapping[str, Any],
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    seed: int,
    trials: int = TRIALS,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """Run retrieve/compose controls on the header-transfer target line."""

    bank = ExternalTemporalProgramBank.load_bank(Path(founding_report["bank"]))
    header_slot = int(founding_report["warm_3cell_3back"]["admission"]["slot"])
    two_back_slot = int(founding_report["warm_3cell_2back"]["admission"]["slot"])
    header_artifact = bank.artifact(header_slot)
    two_back_artifact = bank.artifact(two_back_slot)
    if header_artifact.program_length != 3 or two_back_artifact.program_length != 2:
        raise RuntimeError("founding bank slots do not match 2-back/3-back depths")
    machine = build_recursive_temporal_program_machine(
        controller_payload, sample=False
    )
    if bank.controller_digest != machine.controller_digest():
        raise ValueError("founding bank targets another controller")

    controls = {
        "wrong_depth": execute_loaded_program(
            machine,
            bank,
            two_back_artifact,
            neural_workshop_directory,
            n_back=3,
            active_cells=3,
            trials=trials,
            seed=seed + 4_000,
        ),
        "missing_history": execute_loaded_program(
            machine,
            bank,
            header_artifact,
            neural_workshop_directory,
            n_back=3,
            active_cells=3,
            trials=trials,
            seed=seed + 4_100,
            intervention=NeuralWorkshopIntervention(
                reset_history_each_tick=True, seed=seed + 4_100
            ),
        ),
        "action_reversed": execute_loaded_program(
            machine,
            bank,
            header_artifact,
            neural_workshop_directory,
            n_back=3,
            active_cells=3,
            trials=trials,
            seed=seed + 4_200,
            intervention=NeuralWorkshopIntervention(
                action="reversed", seed=seed + 4_200
            ),
        ),
        # Learner-visible reward shuffle is recorded, not gated. Frozen
        # execution still scores against the private verifier.
        "reward_shuffled": execute_loaded_program(
            machine,
            bank,
            header_artifact,
            neural_workshop_directory,
            n_back=3,
            active_cells=3,
            trials=trials,
            seed=seed + 4_300,
            intervention=NeuralWorkshopIntervention(
                reward="shuffled", seed=seed + 4_300
            ),
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "controls.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True) + "\n"
    )
    flags = founding_control_flags(
        founding_report, controls, threshold=threshold
    )
    return {"controls": controls, "flags": flags}


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
    source_bank_sha256: str,
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
            source_bank_sha256=source_bank_sha256,
            controller_sha256=controller_sha256,
            seeds=seeds,
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
    source_bank_path: Path,
    neural_workshop_directory: Path,
    output_directory: Path,
    *,
    controller_artifact: Path,
    seeds: tuple[int, ...] = HOLDOUT_SEEDS,
    claim_holdout: bool = False,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run the unused holdout population against the frozen gates."""

    assert_unused_holdout_seeds(seeds)
    source_digest = require_source_bank(source_bank_path)
    output_directory.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).parents[2]
    commit = git_commit or current_git_commit(repository)
    controller_digest = sha256_file(controller_artifact)
    started = time.perf_counter()
    seed_rows: list[dict[str, Any]] = []
    replicate_metrics: list[dict[str, float]] = []
    flag_rows: list[dict[str, bool]] = []
    artifact_hashes = {
        "source_bank": source_digest,
        "controller": controller_digest,
        "campaign_runner": sha256_file(Path(__file__)),
    }

    for seed in seeds:
        seed_dir = output_directory / f"seed{seed}"
        founding = run_autonomous_founding(
            controller_payload,
            source_bank_path,
            neural_workshop_directory,
            seed_dir,
            trials=TRIALS,
            threshold=THRESHOLD,
            minimum_bits=MINIMUM_BITS,
            seed=seed,
        )
        control_bundle = run_founding_controls(
            controller_payload,
            founding,
            neural_workshop_directory,
            seed_dir,
            seed=seed,
            trials=TRIALS,
            threshold=THRESHOLD,
        )
        metrics = extract_replicate_metrics(founding)
        replicate_metrics.append(metrics)
        flag_rows.append(control_bundle["flags"])
        seed_report = {
            "seed": seed,
            "founding": founding["founding"],
            "metrics": metrics,
            "control_flags": control_bundle["flags"],
            "controls": control_bundle["controls"],
            "bank_sha256": founding["bank_sha256"],
            "wall_seconds": founding["wall_seconds"],
        }
        (seed_dir / "seed_summary.json").write_text(
            json.dumps(seed_report, indent=2, sort_keys=True) + "\n"
        )
        artifact_hashes[f"seed{seed}_report"] = sha256_file(seed_dir / "report.json")
        seed_rows.append(seed_report)

    controls = combine_control_flags(tuple(flag_rows))
    header_gate = header_transfer_gate()
    depth_gate = first_time_depth_gate()
    header_evidence = build_promotion_evidence(
        header_gate,
        replicate_metrics=tuple(replicate_metrics),
        controls=controls,
        artifact_hashes=artifact_hashes,
        git_commit=commit,
        source_bank_sha256=source_digest,
        controller_sha256=controller_digest,
        seeds=seeds,
    )
    depth_evidence = build_promotion_evidence(
        depth_gate,
        replicate_metrics=tuple(replicate_metrics),
        controls=controls,
        artifact_hashes=artifact_hashes,
        git_commit=commit,
        source_bank_sha256=source_digest,
        controller_sha256=controller_digest,
        seeds=seeds,
    )
    ledger_path = output_directory / "holdout-ledger.jsonl"
    header_decision = evaluate_promotion(header_gate, header_evidence)
    depth_decision = evaluate_promotion(depth_gate, depth_evidence)
    claimed = False
    if claim_holdout:
        ledger = HoldoutLedger(ledger_path)
        ledger.claim(
            HOLDOUT_ID,
            promotion_manifest_digest(seeds=seeds),
            HOLDOUT_ATTEMPT_ID,
        )
        claimed = True
        header_decision = evaluate_promotion(
            header_gate, header_evidence, holdout_ledger=ledger
        )
        depth_decision = evaluate_promotion(
            depth_gate, depth_evidence, holdout_ledger=ledger
        )
        write_promotion_record(
            output_directory / "promotion_header_transfer.json",
            header_gate,
            header_evidence,
            header_decision,
            holdout_ledger=ledger,
        )
        write_promotion_record(
            output_directory / "promotion_first_time_depth.json",
            depth_gate,
            depth_evidence,
            depth_decision,
            holdout_ledger=ledger,
        )
    else:
        write_promotion_record(
            output_directory / "promotion_header_transfer.json",
            header_gate,
            header_evidence,
            header_decision,
        )
        write_promotion_record(
            output_directory / "promotion_first_time_depth.json",
            depth_gate,
            depth_evidence,
            depth_decision,
        )

    campaign = {
        "schema": "neural-computer.founding-holdout-campaign.v1",
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL,
        "holdout_id": HOLDOUT_ID,
        "holdout_claimed": claimed,
        "seeds": list(seeds),
        "source_bank_sha256": source_digest,
        "controller_sha256": controller_digest,
        "git_commit": commit,
        "development_manifest_digest": development_manifest_digest(),
        "promotion_manifest_digest": promotion_manifest_digest(seeds=seeds),
        "configuration_digest": configuration_digest(
            source_bank_sha256=source_digest,
            controller_sha256=controller_digest,
            seeds=seeds,
        ),
        "controls": controls,
        "replicates": seed_rows,
        "header_transfer": header_decision.canonical(),
        "first_time_depth": depth_decision.canonical(),
        "wall_seconds": time.perf_counter() - started,
    }
    (output_directory / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument(
        "--source-bank",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/neural_workshop_instruction_route_seed81017.bank"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_founding_holdout_2026-08-15",
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
        help="consume the one-use holdout lease after the unused seeds finish",
    )
    parser.add_argument(
        "--controls-only",
        type=Path,
        default=None,
        help="run controls against an existing founding report.json (development only)",
    )
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args()
    controller = load_temporal_controller_artifact(arguments.controller_artifact)
    if arguments.controls_only is not None:
        report_path = arguments.controls_only
        if report_path.is_dir():
            report_path = report_path / "report.json"
        founding = json.loads(report_path.read_text(encoding="utf-8"))
        seed = arguments.seed
        if seed is None:
            raise ValueError("--controls-only needs --seed for the control offsets")
        bundle = run_founding_controls(
            controller,
            founding,
            arguments.neural_workshop,
            report_path.parent,
            seed=seed,
        )
        print(json.dumps(bundle, indent=2, sort_keys=True))
        return
    campaign = run_holdout_campaign(
        controller,
        arguments.source_bank,
        arguments.neural_workshop,
        arguments.output_dir,
        controller_artifact=arguments.controller_artifact,
        claim_holdout=arguments.claim_holdout,
    )
    print(json.dumps(campaign, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
