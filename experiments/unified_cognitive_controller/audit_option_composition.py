"""Causal, independent, and retention audits for a learned latent option."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import torch

from .probe_requery_operation import ranked_requery_batch
from .train import evaluate, seed_everything
from .train_option_composition_race import (
    OptionValueHead,
    champion_actions,
    metrics,
    option_physical_actions,
)
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import (
    _load_head,
    paired_ips_improvement,
)
from .verified_skill_store import VerifiedSkillStore


def load_option(path: Path, device: torch.device) -> OptionValueHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "option-composition-head-v1":
        raise ValueError("unsupported option checkpoint")
    head = OptionValueHead(
        int(payload["input_width"]), int(payload["hidden"])).to(device)
    head.load_state_dict(payload["state_dict"])
    return head


def option_skill_payload(head: OptionValueHead) -> dict[str, object]:
    return {
        "head_kind": "option_composition",
        "input_width": int(head.network[0].normalized_shape[0]),
        "hidden": int(head.network[1].out_features),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in head.state_dict().items()},
    }


def option_from_skill_payload(
        payload: dict[str, object],
        device: torch.device | str) -> OptionValueHead:
    if payload.get("head_kind") != "option_composition":
        raise ValueError("unsupported option skill payload")
    head = OptionValueHead(
        int(payload["input_width"]), int(payload["hidden"])).to(device)
    head.load_state_dict(payload["state_dict"])
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--option-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skill-store", type=Path)
    parser.add_argument("--parent-skill-id")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8054)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=2040)
    parser.add_argument("--confirmation-bits", type=int, default=2400)
    parser.add_argument("--capacity", type=int, default=6)
    args = parser.parse_args()
    if bool(args.skill_store) != bool(args.parent_skill_id):
        raise ValueError(
            "--skill-store and --parent-skill-id must be supplied together")
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
    option = load_option(args.option_checkpoint, device)
    costs = torch.tensor([0.0, 0.01, 0.02], device=device)

    streams = []
    for stream in range(args.streams):
        stream_seed = args.seed * 1_000_000 + stream
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.contexts, capacity=args.capacity,
            seed=stream_seed, device=device, write_threshold=0.5,
            candidate_count=3, include_rank_features=True)
        utilities = outcomes - costs
        champion_physical = champion_actions(champion, features)
        option_physical = option_physical_actions(
            option, champion, features)
        generator = torch.Generator(device=device).manual_seed(
            stream_seed + 90_000_000)
        permutation = torch.randperm(
            args.contexts, generator=generator, device=device)
        shuffled_physical = option_physical_actions(
            option, champion, features[permutation])
        use_new = option(features).bool()
        reversed_physical = torch.where(
            ~use_new, torch.full_like(champion_physical, 2),
            champion_physical)
        streams.append({
            "stream": stream,
            "seed": stream_seed,
            "champion": metrics(champion_physical, utilities),
            "option": metrics(option_physical, utilities),
            "feature_shuffled": metrics(shuffled_physical, utilities),
            "option_reversed": metrics(reversed_physical, utilities),
        })

    # Promotion evidence sees only a randomized attempted option and its scalar
    # outcome. Both physical outcomes remain private from the promotion rule.
    features, outcomes, _ = ranked_requery_batch(
        controller, count=args.confirmation_bits, capacity=args.capacity,
        seed=args.seed + 95_000_000, device=device, write_threshold=0.5,
        candidate_count=3, include_rank_features=True)
    utilities = outcomes - costs
    old = champion_actions(champion, features)
    attempted = torch.randint(
        0, 2, (args.confirmation_bits,), device=device,
        generator=torch.Generator(device=device).manual_seed(
            args.seed + 96_000_000))
    attempted_physical = torch.where(
        attempted.bool(), torch.full_like(attempted, 2), old)
    observed = utilities.gather(
        1, attempted_physical[:, None]).squeeze(1)
    candidate_options = option(features)
    evidence = paired_ips_improvement(
        torch.zeros_like(attempted), candidate_options, attempted, observed,
        propensity=0.5)

    restored = load_option(args.option_checkpoint, device)
    reload_exact = torch.equal(
        option(features), restored(features))
    binary = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 97_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 98_000_000, device=device,
        task="four_rule", feedback_trials=2)

    def mean(stage: str, metric: str) -> float:
        return sum(
            row[stage][metric] for row in streams) / len(streams)

    champion_utility = mean("champion", "verified_utility")
    option_utility = mean("option", "verified_utility")
    shuffled_utility = mean("feature_shuffled", "verified_utility")
    reversed_utility = mean("option_reversed", "verified_utility")
    gate = {
        "positive_independent_lower_95": evidence["lower_95"] > 0,
        "mean_gain_at_least_2_points":
            option_utility >= champion_utility + 0.02,
        "every_stream_improves": all(
            row["option"]["verified_utility"]
            > row["champion"]["verified_utility"]
            for row in streams),
        "feature_shuffle_degrades_by_2_points":
            shuffled_utility <= option_utility - 0.02,
        "option_reversal_degrades_by_2_points":
            reversed_utility <= option_utility - 0.02,
        "checkpoint_reload_exact": reload_exact,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    core_accepted = all(gate.values())
    persistence = None
    if args.skill_store is not None and core_accepted:
        store = VerifiedSkillStore(args.skill_store)
        # Loading first proves the declared parent exists and is hash-valid.
        store.load(args.parent_skill_id, device=device)
        checkpoint = torch.load(
            args.option_checkpoint, map_location="cpu", weights_only=False)
        context_key = torch.nn.functional.normalize(
            features.mean(0), dim=0)
        child_id = store.commit(
            option_skill_payload(option),
            context_key=context_key,
            lower_confidence_bound=evidence["lower_95"],
            verifier_bits=(
                int(checkpoint["verifier_bits"])
                + args.confirmation_bits),
            parent_id=args.parent_skill_id,
            provenance={
                "kind": "verified_option_composition",
                "training_seed": checkpoint["training_seed"],
                "training_verifier_bits": checkpoint["verifier_bits"],
                "replay_updates": checkpoint["replay_updates"],
                "audit_seed": args.seed,
            })
        fresh_store = VerifiedSkillStore(args.skill_store)
        loaded = fresh_store.load(child_id, device=device)
        restored_skill = option_from_skill_payload(
            loaded["payload"], device)
        exact = torch.equal(option(features), restored_skill(features))
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
            "entries": fresh_store.entries(),
        }
        gate["persistent_skill_committed"] = all((
            exact, detected, parent_survives))
    elif args.skill_store is not None:
        gate["persistent_skill_committed"] = False
    gate["accepted"] = all(gate.values())
    report = {
        "schema": "option-composition-audit-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "option_checkpoint": str(args.option_checkpoint),
            "report": str(args.report),
            "skill_store": (
                str(args.skill_store)
                if args.skill_store is not None else None),
        },
        "learner_visible_confirmation": [
            "seven_generic_rank_statistics", "randomized_attempted_option",
            "attempted_option_scalar_outcome",
        ],
        "hidden_from_promotion": [
            "unattempted_outcome", "oracle_action", "correct_answer",
            "private_stream_metrics",
        ],
        "means": {
            "champion_utility": champion_utility,
            "option_utility": option_utility,
            "feature_shuffled_utility": shuffled_utility,
            "option_reversed_utility": reversed_utility,
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
                args.streams * args.contexts * 3,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
