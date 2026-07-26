"""Audit and persist third-generation (five-action) option composition."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from .audit_fourth_option_composition import (
    load_router as load_four_router,
    router_from_skill_payload,
    router_skill_payload,
)
from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .train import evaluate, seed_everything
from .train_fifth_option_composition_race import (
    five_action_hierarchy,
    four_action_hierarchy,
    metrics,
)
from .train_option_composition_race import OptionValueHead
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import (
    _load_head,
    paired_ips_improvement,
)
from .verified_skill_store import VerifiedSkillStore


def load_fifth_router(path: Path, device: torch.device) -> OptionValueHead:
    return load_generation_router(
        path, device, schema="fifth-option-router-v1")


def load_generation_router(
        path: Path, device: torch.device, *,
        schema: str) -> OptionValueHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported router checkpoint: expected {schema}")
    router = OptionValueHead(
        int(payload["input_width"]), int(payload["hidden"])).to(device)
    router.load_state_dict(payload["state_dict"])
    return router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--three-option", type=Path, required=True)
    parser.add_argument("--four-router", type=Path, required=True)
    parser.add_argument("--fifth-router", type=Path, required=True)
    parser.add_argument(
        "--sixth-router", type=Path,
        help="When present, audit this router over the fifth-router parent.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skill-store", type=Path, required=True)
    parser.add_argument("--parent-skill-id", required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8085)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=2040)
    parser.add_argument("--confirmation-bits", type=int, default=2400)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--retention-count", type=int, default=512)
    args = parser.parse_args()
    for count in (args.contexts, args.confirmation_bits):
        if count % args.capacity:
            raise ValueError("context counts must divide by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    champion = _load_head(args.champion_head, device)
    option3 = load_option(args.three_option, device)
    router4 = load_four_router(args.four_router, device)
    router5 = load_fifth_router(args.fifth_router, device)
    router_new = (
        load_generation_router(
            args.sixth_router, device, schema="sixth-option-router-v1")
        if args.sixth_router is not None else router5)
    candidate_count = 6 if args.sixth_router is not None else 5
    new_action = candidate_count - 1
    router_width = int(router_new.network[0].normalized_shape[0])
    costs = torch.arange(
        candidate_count, device=device, dtype=torch.float32) * 0.01

    @torch.no_grad()
    def inherited_actions(features: torch.Tensor) -> torch.Tensor:
        if args.sixth_router is None:
            return four_action_hierarchy(
                router4, option3, champion, features)
        return five_action_hierarchy(
            router5, router4, option3, champion, features)

    @torch.no_grad()
    def composed_actions(features: torch.Tensor) -> torch.Tensor:
        old = inherited_actions(features)
        use_new = router_new(features[:, :router_width]).bool()
        return torch.where(
            use_new, torch.full_like(old, new_action), old)

    streams = []
    for stream in range(args.streams):
        stream_seed = args.seed * 1_000_000 + stream
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.contexts, capacity=args.capacity,
            seed=stream_seed, device=device, write_threshold=0.5,
            candidate_count=candidate_count, include_rank_features=True)
        utilities = outcomes - costs
        old = inherited_actions(features)
        composed = composed_actions(features)
        generator = torch.Generator(device=device).manual_seed(
            stream_seed + 90_000_000)
        permutation = torch.randperm(
            args.contexts, generator=generator, device=device)
        shuffled_features = features.clone()
        shuffled_features[:, :router_width] = features[
            permutation, :router_width]
        shuffled_use_fifth = router_new(
            shuffled_features[:, :router_width]).bool()
        shuffled = torch.where(
            shuffled_use_fifth,
            torch.full_like(old, new_action), old)
        use_fifth = router_new(features[:, :router_width]).bool()
        reversed_actions = torch.where(
            ~use_fifth, torch.full_like(old, new_action), old)
        streams.append({
            "stream": stream,
            "seed": stream_seed,
            "previous_hierarchy": metrics(old, utilities, new_action),
            "composition": metrics(composed, utilities, new_action),
            "router_features_shuffled":
                metrics(shuffled, utilities, new_action),
            "router_reversed":
                metrics(reversed_actions, utilities, new_action),
        })

    features, outcomes, _ = ranked_requery_batch(
        controller, count=args.confirmation_bits, capacity=args.capacity,
        seed=args.seed + 95_000_000, device=device, write_threshold=0.5,
        candidate_count=candidate_count, include_rank_features=True)
    utilities = outcomes - costs
    old = inherited_actions(features)
    attempted = torch.randint(
        0, 2, (args.confirmation_bits,), device=device,
        generator=torch.Generator(device=device).manual_seed(
            args.seed + 96_000_000))
    attempted_physical = torch.where(
        attempted.bool(), torch.full_like(old, new_action), old)
    observed = utilities.gather(
        1, attempted_physical[:, None]).squeeze(1)
    candidate_options = router_new(features[:, :router_width])
    evidence = paired_ips_improvement(
        torch.zeros_like(attempted), candidate_options,
        attempted, observed, propensity=0.5)

    audited_checkpoint = (
        args.sixth_router
        if args.sixth_router is not None else args.fifth_router)
    restored = load_generation_router(
        audited_checkpoint, device,
        schema=(
            "sixth-option-router-v1"
            if args.sixth_router is not None
            else "fifth-option-router-v1"))
    reload_exact = torch.equal(
        router_new(features[:, :router_width]),
        restored(features[:, :router_width]))
    binary = evaluate(
        controller, count=args.retention_count, trials=6,
        seed=args.seed + 97_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        controller, count=args.retention_count, trials=6,
        seed=args.seed + 98_000_000, device=device,
        task="four_rule", feedback_trials=2)

    def mean(stage: str) -> float:
        return sum(
            row[stage]["verified_utility"]
            for row in streams) / len(streams)

    old_utility = mean("previous_hierarchy")
    composed_utility = mean("composition")
    shuffled_utility = mean("router_features_shuffled")
    reversed_utility = mean("router_reversed")
    gate = {
        "positive_independent_lower_95": evidence["lower_95"] > 0,
        "mean_gain_at_least_2_points":
            composed_utility >= old_utility + 0.02,
        "every_stream_improves": all(
            row["composition"]["verified_utility"]
            > row["previous_hierarchy"]["verified_utility"]
            for row in streams),
        "router_feature_shuffle_degrades_by_2_points":
            shuffled_utility <= composed_utility - 0.02,
        "router_reversal_degrades_by_2_points":
            reversed_utility <= composed_utility - 0.02,
        "checkpoint_reload_exact": reload_exact,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    core_accepted = all(gate.values())
    persistence = None
    if core_accepted:
        store = VerifiedSkillStore(args.skill_store)
        store.load(args.parent_skill_id, device=device)
        checkpoint = torch.load(
            audited_checkpoint, map_location="cpu", weights_only=False)
        context_key = torch.nn.functional.normalize(
            features.mean(0), dim=0)
        child_id = store.commit(
            router_skill_payload(router_new),
            context_key=context_key,
            lower_confidence_bound=evidence["lower_95"],
            verifier_bits=(
                int(checkpoint["verifier_bits"])
                + args.confirmation_bits),
            parent_id=args.parent_skill_id,
            provenance={
                "kind": "verified_hierarchical_option_composition",
                "generation": (
                    4 if args.sixth_router is not None else 3),
                "training_seed": checkpoint["training_seed"],
                "training_verifier_bits": checkpoint["verifier_bits"],
                "feedback_mode": checkpoint.get("feedback_mode", "bandit"),
                "replay_updates": checkpoint["replay_updates"],
                "actual_optimizer_updates": checkpoint.get(
                    "actual_optimizer_updates"),
                "actual_replayed_examples": checkpoint.get(
                    "actual_replayed_examples"),
                "adaptive_replay_loss": checkpoint.get(
                    "adaptive_replay_loss"),
                "audit_seed": args.seed,
            })
        fresh = VerifiedSkillStore(args.skill_store)
        loaded = fresh.load(child_id, device=device)
        restored_skill = router_from_skill_payload(
            loaded["payload"], device)
        exact = torch.equal(
            router_new(features[:, :router_width]),
            restored_skill(features[:, :router_width]))
        with tempfile.TemporaryDirectory() as directory:
            corrupt_root = Path(directory) / "store"
            shutil.copytree(args.skill_store, corrupt_root)
            corrupt = VerifiedSkillStore(corrupt_root)
            entry = next(
                row for row in corrupt.entries()
                if row["skill_id"] == child_id)
            child_path = corrupt_root / entry["file"]
            child_path.write_bytes(child_path.read_bytes() + b"corrupt")
            detected = False
            try:
                corrupt.load(child_id)
            except ValueError:
                detected = True
            parent_survives = (
                corrupt.load(args.parent_skill_id)["schema"]
                == "verified-latent-skill-v1")
        persistence = {
            "root": str(args.skill_store),
            "parent_id": args.parent_skill_id,
            "child_id": child_id,
            "reload_exact": exact,
            "corruption_detected": detected,
            "parent_survives_child_corruption": parent_survives,
            "entries": fresh.entries(),
        }
        gate["persistent_skill_committed"] = all((
            exact, detected, parent_survives))
    else:
        gate["persistent_skill_committed"] = False
    gate["accepted"] = all(gate.values())
    report = {
        "schema": (
            "sixth-option-composition-audit-v1"
            if args.sixth_router is not None
            else "fifth-option-composition-audit-v1"),
        "configuration": {
            **vars(args),
            **{
                key: str(getattr(args, key))
                for key in (
                    "parent_checkpoint", "selected_prefix",
                    "champion_head", "three_option", "four_router",
                    "fifth_router", "sixth_router",
                    "report", "skill_store")
            },
        },
        "means": {
            "previous_hierarchy_utility": old_utility,
            "composition_utility": composed_utility,
            "router_features_shuffled_utility": shuffled_utility,
            "router_reversed_utility": reversed_utility,
        },
        "independent_confirmation": evidence,
        "streams": streams,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "verified_skill_store": persistence,
        "gate": gate,
        "accounting": {
            "training_verifier_bits": 0,
            "fresh_confirmation_verifier_bits": args.confirmation_bits,
            "private_audit_both_action_bits":
                args.streams * args.contexts * candidate_count,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
