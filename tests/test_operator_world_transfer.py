from __future__ import annotations

from experiments.brainworkshop_canonical.operator_world_transfer import (
    ACTION_COUNT,
    PLACE_COUNT,
    SOURCE_WORLD_SEED,
    TARGET_WORLD_SEED,
    VerifiedOperatorBundle,
    VerifiedPlanningOperatorStager,
    build_raw_successor_artifact,
    corrupted_bundle,
    irrelevant_bundle,
    observe_transition,
    run_arm,
    run_transfer,
    sample_ring_world,
    verified_bundle,
)
from experiments.brainworkshop_canonical.world_model import WorldModel


def test_world_change_does_not_change_the_protocol_shape() -> None:
    source = sample_ring_world(SOURCE_WORLD_SEED)
    target = sample_ring_world(TARGET_WORLD_SEED)
    assert source.transitions != target.transitions
    assert len(source.transitions) == ACTION_COUNT
    assert all(len(row) == PLACE_COUNT for row in source.transitions)
    assert source.digest != target.digest


def test_operator_bundle_contains_no_world_specific_successor_data() -> None:
    bundle = verified_bundle(world_seed=SOURCE_WORLD_SEED)
    payload = bundle.payload()
    assert bundle.validate() is bundle
    assert "transitions" not in payload
    assert "policies" not in payload
    assert "psis" not in payload
    assert payload["source_family"] == "ring-v1"


def test_irrelevant_and_corrupted_artifacts_are_distinct() -> None:
    assert irrelevant_bundle().digest != corrupted_bundle(world_seed=SOURCE_WORLD_SEED).digest
    invalid = VerifiedOperatorBundle(source_family="other-family")
    try:
        invalid.validate()
    except ValueError as error:
        assert "unrelated" in str(error)
    else:
        raise AssertionError("an unrelated operator family was accepted")


def test_raw_successor_artifact_is_bound_to_its_source_world() -> None:
    source = sample_ring_world(SOURCE_WORLD_SEED)
    target = sample_ring_world(TARGET_WORLD_SEED)
    artifact = build_raw_successor_artifact(source)
    assert artifact.source_world_digest == source.digest
    assert artifact.source_world_digest != target.digest
    assert artifact.digest


def test_reusable_update_operator_rebuilds_after_a_contradiction() -> None:
    model = WorldModel(PLACE_COUNT, ACTION_COUNT)
    bundle = verified_bundle(world_seed=SOURCE_WORLD_SEED)
    observe_transition(model, 0, 0, 1, 0, bundle=bundle)
    observe_transition(model, 0, 0, 2, 0, bundle=bundle)
    assert model.successor(0, 0) == 2
    assert model.known_cells == 1

    model = WorldModel(PLACE_COUNT, ACTION_COUNT)
    observe_transition(model, 0, 0, 1, 0, bundle=irrelevant_bundle())
    observe_transition(model, 0, 0, 2, 0, bundle=irrelevant_bundle())
    assert model.successor(0, 0) == 1
    assert model.counts[0][0] == {1: 1, 2: 1}


def test_operator_stager_skips_missing_evidence_and_quarantines_reversal() -> None:
    bundle = verified_bundle(world_seed=SOURCE_WORLD_SEED)
    stager = VerifiedPlanningOperatorStager(
        threshold=0.75,
        min_observations=2,
        min_stable_observations=2,
    )
    stager.observe(bundle, 0.0, eligible=False)
    assert stager.status(bundle).observations == 0
    stager.observe(bundle, 1.0)
    stager.observe(bundle, 1.0)
    admission = stager.admit_verified(bundle, lambda candidate: candidate.digest == bundle.digest)
    assert admission.accepted
    stager.observe(bundle, 0.0)
    status = stager.status(bundle)
    assert status.quarantined
    assert not status.accepted
    assert status.observations == 2


def test_operator_stager_retention_probe_is_a_real_gate() -> None:
    bundle = verified_bundle(world_seed=SOURCE_WORLD_SEED)
    stager = VerifiedPlanningOperatorStager(
        threshold=0.75,
        min_observations=2,
        min_stable_observations=2,
    )
    stager.observe(bundle, 1.0)
    stager.observe(bundle, 1.0)
    rejected = stager.admit_verified(bundle, lambda candidate: False)
    assert not rejected.accepted
    accepted = stager.admit_verified(bundle, lambda candidate: True)
    assert accepted.accepted


def test_reusable_operator_reaches_a_stable_prefix_before_fresh() -> None:
    target = sample_ring_world(TARGET_WORLD_SEED)
    reusable = run_arm(
        target,
        mode="reusable",
        bundle=verified_bundle(world_seed=SOURCE_WORLD_SEED),
        artifact=None,
        seed=41,
        training_episodes=8,
        evaluation_episodes=3,
    )
    fresh = run_arm(
        target,
        mode="fresh",
        bundle=None,
        artifact=None,
        seed=41,
        training_episodes=8,
        evaluation_episodes=3,
    )
    assert reusable["stable_bits_to_threshold"] is not None
    assert fresh["stable_bits_to_threshold"] is not None
    assert reusable["stable_bits_to_threshold"] <= fresh["stable_bits_to_threshold"]
    assert reusable["unique_logical_lifetimes"] == fresh["unique_logical_lifetimes"]
    assert reusable["optimizer_updates"] == fresh["optimizer_updates"] == 0


def test_raw_successor_and_corrupted_operator_do_not_claim_stable_transfer() -> None:
    target = sample_ring_world(TARGET_WORLD_SEED)
    raw = run_arm(
        target,
        mode="raw_successor",
        bundle=None,
        artifact=build_raw_successor_artifact(sample_ring_world(SOURCE_WORLD_SEED)),
        seed=41,
        training_episodes=8,
        evaluation_episodes=3,
    )
    corrupted = run_arm(
        target,
        mode="corrupted",
        bundle=corrupted_bundle(world_seed=SOURCE_WORLD_SEED),
        artifact=None,
        seed=41,
        training_episodes=8,
        evaluation_episodes=3,
    )
    assert raw["stable_bits_to_threshold"] is None
    assert corrupted["stable_bits_to_threshold"] is None


def test_transfer_report_keeps_arm_accounting_matched(tmp_path) -> None:
    report = run_transfer(
        tmp_path,
        replicates=1,
        training_episodes=8,
        evaluation_episodes=3,
    )
    assert report["claim_status"] == "development_diagnostic"
    assert report["transfer_ratio_against_fresh_learner"] <= 1.0
    accounting = report["accounting"]
    assert {
        accounting[arm]["unique_verifier_bits"] for arm in accounting
    } == {224}
    assert all(accounting[arm]["optimizer_updates"] == 0 for arm in accounting)
