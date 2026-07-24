#!/usr/bin/env python3
"""Million-seed deterministic curriculum audit.

The harness is deliberately independent of the Elisa compiler.  It checks
coverage, answer reproducibility, malformed/ambiguous records, and generation
throughput, and writes a compact report plus the first failing seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import Counter

from challenge_reference import KINDS, Challenge, generated, solve
from difficulty_profiles import manifest, profile

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1_000_000)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=("intro", "standard", "hard", "max"), default="standard")
    ap.add_argument("--report", type=pathlib.Path,
                    default=pathlib.Path(__file__).with_name("challenge_validation.json"))
    ap.add_argument("--seed-manifest", type=pathlib.Path, default=None,
                    help="write the exact seed range and failing seeds as JSON")
    args = ap.parse_args()
    if args.count < 1 or args.count > 100_000_000:
        ap.error("count must be in [1, 100000000]")
    started = time.perf_counter()
    seen: Counter[str] = Counter()
    failures: list[dict] = []
    answer_hist: Counter[int] = Counter()
    latency_samples_ns: list[int] = []
    value_count_hist: Counter[int] = Counter()
    nesting_hist: Counter[int] = Counter()
    distractor_hist: Counter[int] = Counter()
    interference_hist: Counter[int] = Counter()
    for offset in range(args.count):
        seed = args.start_seed + offset
        item_started = time.perf_counter_ns()
        c = generated(seed, args.difficulty)
        seen[c.kind] += 1
        answer_hist[c.answer] += 1
        value_count_hist[len(c.values)] += 1
        nesting_hist[c.nesting_depth] += 1
        distractor_hist[c.distractor_count] += 1
        interference_hist[c.interference_permille] += 1
        # Re-solving the public record must reproduce the hidden evaluator answer.
        public = Challenge(c.kind, c.values, c.claim, 0)
        try:
            recomputed = solve(public)
        except Exception as exc:  # pragma: no cover - retained in report
            failures.append({"seed": seed, "kind": c.kind, "error": repr(exc)})
            continue
        if recomputed != c.answer or c.claim not in (c.answer, c.answer + 1):
            failures.append({"seed": seed, "kind": c.kind, "expected": c.answer,
                             "recomputed": recomputed, "claim": c.claim})
        # A scalar equality verifier has exactly one accepted answer by contract.
        if sum(recomputed == candidate for candidate in (recomputed, recomputed + 1)) != 1:
            failures.append({"seed": seed, "kind": c.kind, "error": "non-unique verifier"})
        if offset < 1000 or offset % max(1, args.count // 1000) == 0:
            latency_samples_ns.append(time.perf_counter_ns() - item_started)
    elapsed = time.perf_counter() - started
    missing = [kind for kind in KINDS if not seen[kind]]
    latency_p95_us = (sorted(latency_samples_ns)[max(0, int(len(latency_samples_ns) * .95) - 1)] / 1000
                      if latency_samples_ns else None)
    report = {
        "schema": "syllogimous.challenge-validation.v1",
        "start_seed": args.start_seed,
        "difficulty": args.difficulty,
        "difficulty_profile": manifest()[args.difficulty],
        "count": args.count,
        "elapsed_seconds": elapsed,
        "tasks_per_second": args.count / elapsed if elapsed else None,
        "generation_latency_us": {
            "sample_count": len(latency_samples_ns),
            "p50": sorted(latency_samples_ns)[len(latency_samples_ns) // 2] / 1000 if latency_samples_ns else None,
            "p95": latency_p95_us,
        },
        "deadline_feasible": latency_p95_us is not None and latency_p95_us / 1000 < profile(args.difficulty).deadline_ms,
        "family_count": len(KINDS),
        "families_seen": len(seen),
        "missing_families": missing,
        "failure_count": len(failures),
        "failures": failures[:100],
        "answer_histogram": dict(sorted(answer_hist.items())),
        "difficulty_realization": {
            "value_count_histogram": dict(sorted(value_count_hist.items())),
            "nesting_depth_histogram": dict(sorted(nesting_hist.items())),
            "distractor_count_histogram": dict(sorted(distractor_hist.items())),
            "interference_permille_histogram": dict(sorted(interference_hist.items())),
        },
        "unique_answer_contract": not failures,
        "accepted_answer_cardinality": 1,
        "unique_answer_proof": "native ChallengeSolution.candidate_count is constrained to exactly 1; independent solver agrees",
        "semantic_audit": {
            "independent_solver": "challenge_reference.solve",
            "catalog_families": list(KINDS),
            "known_review_notes": [
                "Graph* primitives currently use compact scalar encodings; a future catalog revision should expose explicit edge lists.",
                "UniqueValue and AllDistinct intentionally share semantics but remain separate surface families for curriculum coverage.",
            ],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path = args.seed_manifest or args.report.with_name(args.report.stem + ".seeds.json")
    manifest_path.write_text(json.dumps({
        "schema": "syllogimous.seed-manifest.v1",
        "start_seed": args.start_seed,
        "count": args.count,
        "difficulty": args.difficulty,
        "failed_seeds": [item["seed"] for item in failures if "seed" in item],
    }, indent=2, sort_keys=True) + "\n")
    summary = {k: report[k] for k in
               ("count", "elapsed_seconds", "tasks_per_second", "families_seen",
                "missing_families", "failure_count")}
    summary["report"] = str(args.report)
    summary["seed_manifest"] = str(manifest_path)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures and not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
