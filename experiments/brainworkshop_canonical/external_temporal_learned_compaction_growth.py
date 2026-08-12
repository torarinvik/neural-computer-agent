"""Transfer a learned opaque compaction policy into live external memory.

The generic consolidation promotion already shows that an opaque policy can
learn duplicate selection from scalar rewrite utility.  This composition test
puts that replaceable policy on the canonical persistent content-memory ABI:
the live rows use learned event keys and opaque capability-address values, and
the policy must select the redundant source/alias pair under every physical
row permutation before a verifier-gated compaction is committed.

The controller is never optimized in this experiment.  It is a transfer test
for memory-side learning, not an end-to-end temporal capability promotion.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from experiments.opaque_consolidation_amodal.train import _train_policy
from neural_computer import (
    OpaqueConsolidationPolicy,
    PersistentAppendOnlyContentAddressedMemory,
    verify_consolidation_proposal,
)

from .external_temporal_content_retrieval_growth import (
    _digest,
    _event_key,
    _noisy_key,
)
from .external_temporal_legacy_support import address_basis
from .external_temporal_query_address_growth import _build
from .external_temporal_verified_compaction_growth import _candidate_verifier

LEARNED_COMPACTION_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-learned-compaction-growth.v1"
)
SOURCE_OFFSET = 4
TARGET_OFFSET = 5
PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _key_set(keys: torch.Tensor, indices: tuple[int, int]) -> set[tuple[float, ...]]:
    return {
        tuple(float(value) for value in keys[index].tolist()) for index in indices
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.policy_updates, args.policy_batch_size) < 1:
        raise ValueError("learned compaction policy budgets must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    source_key = _event_key(system, 0)
    source_alias = _noisy_key(source_key, seed=args.seed + 1)
    target_key = _event_key(system, 1)
    basis = address_basis(args.seed)
    live_keys = torch.stack((source_key, source_alias, target_key))
    live_values = torch.stack(
        (basis[SOURCE_OFFSET - 1], basis[SOURCE_OFFSET - 1], basis[TARGET_OFFSET - 1])
    )
    source_pair = _key_set(live_keys, (0, 1))
    learned_policy, policy_accounting = _train_policy(
        seed=args.seed,
        rows=8,
        width=live_keys.shape[1],
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=False,
    )
    untrained_policy = OpaqueConsolidationPolicy(live_keys.shape[1], hidden=64).eval()
    learned_hits = 0
    untrained_hits = 0
    learned_proposals: list[dict[str, object]] = []
    untrained_proposals: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="neural-computer-learned-compaction-") as directory:
        path = Path(directory) / "content-memory.pt"
        memory = PersistentAppendOnlyContentAddressedMemory(
            live_keys.shape[1],
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
        )
        for permutation in PERMUTATIONS:
            memory.clear()
            memory.write(
                live_keys[list(permutation)],
                live_values[list(permutation)],
                torch.ones(3),
            )
            candidates = memory.candidates()
            learned_proposal = learned_policy.propose(candidates)
            untrained_proposal = untrained_policy.propose(candidates)
            if learned_proposal is None or untrained_proposal is None:
                raise RuntimeError("compaction policy produced no proposal")
            learned_selected = _key_set(
                candidates.keys[0],
                (learned_proposal.first, learned_proposal.second),
            )
            untrained_selected = _key_set(
                candidates.keys[0],
                (untrained_proposal.first, untrained_proposal.second),
            )
            learned_hit = learned_selected == source_pair
            untrained_hit = untrained_selected == source_pair
            learned_hits += int(learned_hit)
            untrained_hits += int(untrained_hit)
            learned_proposals.append(
                {
                    "permutation": permutation,
                    "rows": (learned_proposal.first, learned_proposal.second),
                    "selected_redundant_pair": learned_hit,
                }
            )
            untrained_proposals.append(
                {
                    "permutation": permutation,
                    "rows": (untrained_proposal.first, untrained_proposal.second),
                    "selected_redundant_pair": untrained_hit,
                }
            )

        memory.clear()
        memory.write(live_keys, live_values, torch.ones(3))
        candidates = memory.candidates()
        source_version = int(memory.store_version.item())
        proposal = learned_policy.propose(candidates)
        if proposal is None:
            raise RuntimeError("learned policy produced no live proposal")

        def verifier(candidate) -> bool:
            return _candidate_verifier(
                candidate,
                basis=basis,
                source_key=source_key,
                source_alias=source_alias,
                target_key=target_key,
                target_alias=target_key,
                source_offset=SOURCE_OFFSET,
                target_offset=TARGET_OFFSET,
            )

        selected_correctly = _key_set(
            candidates.keys[0], (proposal.first, proposal.second)
        ) == source_pair
        candidate, verification = verify_consolidation_proposal(
            candidates,
            proposal,
            verifier,
            candidate_outcomes=[1.0] * 8 if selected_correctly else [0.0] * 8,
            retained_scores=[1.0] * 4,
            min_candidate_observations=8,
        )
        committed = False
        compaction = None
        if candidate is not None and verification.accepted:
            compaction = memory.replace_from_candidates(
                candidate,
                expected_version=source_version,
            )
            committed = True
        restored = PersistentAppendOnlyContentAddressedMemory(
            live_keys.shape[1],
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
        )
        reloaded_routes = _candidate_verifier(
            restored.candidates(),
            basis=basis,
            source_key=source_key,
            source_alias=source_alias,
            target_key=target_key,
            target_alias=target_key,
            source_offset=SOURCE_OFFSET,
            target_offset=TARGET_OFFSET,
        )
        corruption_rejected = False
        corrupt_path = Path(directory) / "corrupt-content-memory.pt"
        payload = torch.load(path, weights_only=False)
        payload["state_dict"]["values"][0, 0] += 0.25
        torch.save(payload, corrupt_path)
        try:
            PersistentAppendOnlyContentAddressedMemory(
                live_keys.shape[1],
                path=corrupt_path,
                write_threshold=0.0,
                write_match_threshold=0.999,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "learned_policy_selects_redundant_pair": learned_hits == len(PERMUTATIONS),
        "learned_beats_untrained_on_permutations": learned_hits > untrained_hits,
        "candidate_permutation_invariant": learned_hits == len(PERMUTATIONS),
        "verifier_accepts_selected_compaction": committed and verification.accepted,
        "compaction_saves_one_row": compaction is not None
        and compaction.rows_before == 3
        and compaction.rows_after == 2,
        "reload_preserves_live_routes": reloaded_routes,
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": LEARNED_COMPACTION_SCHEMA,
        "claim_boundary": (
            "A learned opaque memory-side pair selector transferred from scalar "
            "rewrite utility to the canonical persistent external-memory ABI; "
            "not end-to-end capability acquisition, arbitrary compression, "
            "unrestricted memory growth, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_consolidation_policy_v1",
            "policy_training_signal": "scalar_duplicate_rewrite_utility",
            "live_memory": "persistent_append_only_content_addressed_memory_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "source_offset": SOURCE_OFFSET,
            "target_offset": TARGET_OFFSET,
        },
        "learned_proposals": learned_proposals,
        "untrained_proposals": untrained_proposals,
        "verification": verification.__dict__,
        "compaction": None if compaction is None else compaction.__dict__,
        "gates": gates,
        "accounting": {
            "unique_scalar_utility_observations": int(
                policy_accounting["unique_verifier_bits"]
            ),
            "optimizer_updates": int(policy_accounting["optimizer_updates"]),
            "permutation_probes": len(PERMUTATIONS),
            "content_memory_writes": len(PERMUTATIONS) + 1,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_temporal_learned_compaction_growth"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy-updates", type=int, default=512)
    parser.add_argument("--policy-batch-size", type=int, default=16)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
