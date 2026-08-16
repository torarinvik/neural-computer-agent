"""Four ways to break a remembered self, and one it should refuse.

Carrying a self model across episodes took identification from 0.47 to 0.92,
which is a large enough jump to be suspicious of. A mechanism that says "the
thing whose future my actions explain is me" has an obvious failure mode --
it can be confidently wrong, and confidence that survives being wrong is worse
than no mechanism at all.

So four controls, each attacking a different assumption, and a fifth property
that matters more than any of them.

**Transplanted.** A self model fitted in one world, used in another. The
dynamics are unrelated, so it should be worth nothing -- and, more importantly,
should not be worth less than nothing by confidently naming the wrong track.

**Reversed.** A self model fitted in one world is evaluated immediately after
the world's dynamics change, before any hindsight or re-fitting. This tests
change refusal, not recovery: no online invalidation mechanism exists yet.

**Poisoned.** Seeded deliberately with the distractor's dynamics instead of the
agent's. This is the self-confirming loop as an experiment rather than as an
accident: if the soft posterior cannot escape a bad start, its advantage over
the hard version is a matter of luck in initialisation.

**Mimicked.** A second marker that responds to the agent's actions *exactly as
the agent does*. There is then no fact of the matter about which one is the
agent -- the two are dynamically identical -- and the only correct behaviour is
to **abstain**. A system that always names something will name one of them with
full confidence, and be right half the time by construction, which is the kind
of number that looks like competence and is not.

Abstention is measured everywhere, as the share of episodes whose posterior is
too flat to call. A mechanism that cannot say "I do not know" is not usable by
anything that has to act on the answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .integrated_navigation import _episode_tracks, collect, weighted_models
from .navigation_environment import NavigationTask, sample_navigation_task
from .object_scene import PLACE_COUNT, render_markers
from .relational_transfer import target_circuit
from .slot_alignment import (
    TrackHistory,
    identify_roles,
    self_log_evidence,
    self_posterior,
)
from .successor_transfer import SlotReader, build_slot_reader

EXPERIMENT_ID = "brainworkshop-self-model-adversarial-2026-08-16"
SELF_MODEL_ADVERSARIAL_SCHEMA = "neural-computer.self-model-adversarial.v1"
DEVELOPMENT_SEED = 41
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
FRAME_SIZE = 36
EPISODE_STEPS = 20
EXPLORE_EPISODES = 40
PASSES = 6
# A posterior flatter than this names nothing. Two tracks at 0.5 each is the
# case the mimic condition manufactures on purpose.
CONFIDENT = 0.75
CONDITIONS = ("honest", "transplanted", "reversed", "poisoned", "mimicked")


def likelihood_only_posterior(
    histories, counts, *, alphabet: int
) -> list[float]:
    """The committed pre-audit posterior, retained as an explicit ablation."""

    if not histories:
        return []
    evidence = [
        self_log_evidence(history, counts, alphabet=alphabet)
        for history in histories
    ]
    best = max(evidence)
    weights = [2.0 ** (value - best) for value in evidence]
    total = sum(weights) or 1.0
    return [value / total for value in weights]


def truth_of(episode) -> int | None:
    """Scoring-side: which track actually followed the agent."""

    trace = episode["trace"]
    width = min((len(row) for row in trace), default=0)
    if not width:
        return None
    return max(
        range(width),
        key=lambda track: sum(
            1
            for step, row in enumerate(trace)
            if row[track] == episode["oracle"][step]
        ),
    )


def prepared(
    reader: SlotReader,
    task: NavigationTask,
    circuit: tuple[int, ...],
    *,
    episodes: int,
    steps: int,
    seed: int,
    cluster_of_place,
):
    """Collect episodes and lay the correspondence beam over each."""

    collected, _ = collect(
        reader,
        task,
        circuit,
        arm="integrated",
        episodes=episodes,
        steps=steps,
        seed=seed,
        cluster_of_place=cluster_of_place,
    )
    for episode in collected:
        trace, histories = _episode_tracks(episode, alphabet=reader.alphabet)
        episode["trace"] = trace
        episode["histories"] = histories
    return collected


def initial_posteriors(collected):
    """Pass zero: one episode's own evidence, flattened to a soft vote."""

    posteriors = []
    for episode in collected:
        width = len(episode["histories"])
        roles = identify_roles(episode["histories"])
        vote = [0.0] * width
        if roles.own is not None:
            vote[roles.own] = 1.0
        elif width:
            vote = [1.0 / width] * width
        posteriors.append(vote)
    return posteriors


def poisoned_posteriors(collected):
    """Start by insisting the agent is whichever track it is not."""

    posteriors = []
    for episode in collected:
        width = len(episode["histories"])
        vote = [0.0] * width
        actual = truth_of(episode)
        wrong = next(
            (index for index in range(width) if index != actual), None
        )
        if wrong is not None:
            vote[wrong] = 1.0
        elif width:
            vote = [1.0 / width] * width
        posteriors.append(vote)
    return posteriors


def refit(
    collected,
    posteriors,
    *,
    reader: SlotReader,
    task: NavigationTask,
    passes: int = PASSES,
    posterior_fn=self_posterior,
    accounting: dict[str, int] | None = None,
):
    """Alternate "which track was me" with "what do I do"."""

    if accounting is not None:
        accounting["replayed_episode_histories"] += len(collected)
    self_counts, _ = weighted_models(
        collected, posteriors, reader=reader, task=task
    )
    for _ in range(int(passes)):
        if accounting is not None:
            # One pass scores every stored history and a second pass rebuilds
            # the weighted model from the same stored histories.
            accounting["replayed_episode_histories"] += 2 * len(collected)
        posteriors = [
            posterior_fn(
                episode["histories"], self_counts, alphabet=reader.alphabet
            )
            for episode in collected
        ]
        self_counts, _ = weighted_models(
            collected, posteriors, reader=reader, task=task
        )
    return posteriors, self_counts


def score(collected, posteriors) -> dict[str, float]:
    """Named correctly, named at all, and named wrongly with confidence."""

    named = right = confident_wrong = 0
    for episode, posterior in zip(collected, posteriors):
        if not posterior:
            continue
        top = max(range(len(posterior)), key=lambda index: posterior[index])
        confident = posterior[top] >= CONFIDENT
        actual = truth_of(episode)
        if confident:
            named += 1
            if top == actual:
                right += 1
            else:
                confident_wrong += 1
    total = max(1, len(collected))
    return {
        "named": named / total,
        "abstained": 1.0 - named / total,
        # Of the episodes it was willing to call, how many it got right.
        "precision": right / named if named else 0.0,
        "confidently_wrong": confident_wrong / total,
    }




def run_self_model_adversarial(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    world_seed: int = DEVELOPMENT_WORLD_SEED,
    tasks: int = 3,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
) -> dict[str, Any]:
    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    reader = build_slot_reader(encoders)
    cluster_of_place = tuple(
        reader.read(render_markers((place, place), size=FRAME_SIZE))[0][1]
        for place in range(PLACE_COUNT)
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    accounting = {"replayed_episode_histories": 0}
    for index in range(tasks):
        task = sample_navigation_task(seed=world_seed + WORLD_SEED_STRIDE * index)
        other = sample_navigation_task(
            seed=world_seed + WORLD_SEED_STRIDE * (index + 100)
        )
        if task is None or other is None:
            continue
        circuit = target_circuit(seed + index)
        collected = prepared(
            reader, task, circuit, episodes=explore_episodes, steps=steps,
            seed=seed + 100 * index, cluster_of_place=cluster_of_place,
        )
        elsewhere = prepared(
            reader, other, circuit, episodes=explore_episodes, steps=steps,
            seed=seed + 100 * index + 7, cluster_of_place=cluster_of_place,
        )

        entry: dict[str, Any] = {"task": index}

        honest, self_counts = refit(
            collected,
            initial_posteriors(collected),
            reader=reader,
            task=task,
            accounting=accounting,
        )
        entry["honest"] = score(collected, honest)

        # A self model fitted somewhere else entirely.
        _, foreign = refit(
            elsewhere,
            initial_posteriors(elsewhere),
            reader=reader,
            task=other,
            accounting=accounting,
        )
        accounting["replayed_episode_histories"] += len(collected)
        transplanted = [
            self_posterior(episode["histories"], foreign, alphabet=reader.alphabet)
            for episode in collected
        ]
        entry["transplanted"] = score(collected, transplanted)

        # The ground moves without notification. Score the already-fitted
        # model on the first episodes from the new world, before any hindsight
        # or re-fitting can make the change look easier than it is.
        accounting["replayed_episode_histories"] += len(elsewhere)
        shifted = [
            self_posterior(
                episode["histories"], self_counts, alphabet=reader.alphabet
            )
            for episode in elsewhere
        ]
        entry["reversed"] = score(elsewhere, shifted)

        # Deliberately started on the wrong track.
        poisoned, _ = refit(
            collected,
            poisoned_posteriors(collected),
            reader=reader,
            task=task,
            accounting=accounting,
        )
        entry["poisoned"] = score(collected, poisoned)

        # Two markers that respond to the agent's actions identically.
        # This is a diagnostic transformation of the already-paid stream, not
        # another interaction falsely counted as unique experience.
        mimicked = copy.deepcopy(collected)
        for episode in mimicked:
            # Both tracks are given the agent's own transitions, so no evidence
            # in the world distinguishes them.
            trace = [tuple([row[0]] * len(row)) for row in episode["trace"]]
            episode["trace"] = trace
            _, histories = _episode_tracks(episode, alphabet=reader.alphabet)
            copies = []
            for _ in range(len(trace[0])):
                duplicate = TrackHistory()
                for symbol, action, following in histories[0].steps:
                    duplicate.observe(symbol, action, following)
                copies.append(duplicate)
            episode["histories"] = copies
        twins, _ = refit(
            mimicked,
            initial_posteriors(mimicked),
            reader=reader,
            task=task,
            accounting=accounting,
        )
        entry["mimicked"] = score(mimicked, twins)

        # The guard and controllability term were added after the first
        # integrated result. Keep the exact earlier posterior beside the new
        # one so an apparent rescue cannot be attributed to an unreported
        # change of baseline.
        old_honest, _ = refit(
            collected,
            initial_posteriors(collected),
            reader=reader,
            task=task,
            posterior_fn=likelihood_only_posterior,
            accounting=accounting,
        )
        old_poisoned, _ = refit(
            collected,
            poisoned_posteriors(collected),
            reader=reader,
            task=task,
            posterior_fn=likelihood_only_posterior,
            accounting=accounting,
        )
        entry["ablation"] = {
            "guarded_honest": entry["honest"],
            "guarded_poisoned": entry["poisoned"],
            "likelihood_only_honest": score(collected, old_honest),
            "likelihood_only_poisoned": score(collected, old_poisoned),
        }
        rows.append(entry)

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the adversarial run mutated AgentBrain.bank")

    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        block = [row[condition] for row in rows if condition in row]
        if not block:
            continue
        summary[condition] = {
            key: sum(item[key] for item in block) / len(block)
            for key in ("named", "abstained", "precision", "confidently_wrong")
        }

    elapsed = time.perf_counter() - started
    ablation_names = tuple(rows[0]["ablation"]) if rows else ()
    ablations = {
        name: {
            key: sum(row["ablation"][name][key] for row in rows) / len(rows)
            for key in ("named", "abstained", "precision", "confidently_wrong")
        }
        for name in ablation_names
    }
    report = {
        "schema": SELF_MODEL_ADVERSARIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": len(rows),
        "explore_episodes": explore_episodes,
        "confident_at": CONFIDENT,
        "conditions": summary,
        "ablations": ablations,
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": elapsed,
        "accounting": {
            # `collect` emits one deterministic scalar per action. Current and
            # alternate worlds are the only distinct interaction streams; all
            # other arms re-read or transform them.
            "unique_verifier_bits": len(rows) * 2 * explore_episodes * steps,
            "unique_logical_lifetimes": len(rows) * 2 * explore_episodes,
            "optimizer_updates": 0,
            "replayed_examples": accounting["replayed_episode_histories"],
            "wall_seconds": elapsed,
            "latency": "not measured separately from the offline diagnostic",
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": "not claimed",
            "transfer_ratio_against_fresh_learner": None,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "self_model_adversarial.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank", type=Path, default=repository / "artifacts/checkpoints/AgentBrain.bank"
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_self_model_adversarial_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED)
    parser.add_argument("--tasks", type=int, default=3)
    arguments = parser.parse_args()
    report = run_self_model_adversarial(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        world_seed=arguments.world_seed,
        tasks=arguments.tasks,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
