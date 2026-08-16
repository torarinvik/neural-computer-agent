"""Break a remembered self, then measure whether it can recover safely.

Carrying a self model across episodes took identification from 0.47 to 0.92,
which is a large enough jump to be suspicious of. A mechanism that says "the
thing whose future my actions explain is me" has an obvious failure mode --
it can be confidently wrong, and confidence that survives being wrong is worse
than no mechanism at all.

The audit has explicit posterior baselines, a causal dynamics reversal, and
near-mimics in addition to the exact symmetry control. The point is not to find
a winning number. It is to make a self model fail closed when its old causal
story no longer applies, and to account for the experience needed to learn a
replacement.

**Transplanted.** A self model fitted in one world, used in another. The
dynamics are unrelated, so it should be worth nothing -- and, more importantly,
should not be worth less than nothing by confidently naming the wrong track.

**Reversed.** A self model is fitted before the world's dynamics change. After
the unannounced change, each episode is scored against the still-live model;
predictive collapse quarantines it, and only then may a post-change model be
rebuilt from the episodes observed since the change. Detection and recovery
are measured per episode, never by refitting over the whole stream.

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

**Near mimics.** Delayed, partially responsive, stochastic, and independently
controlled distractors test whether abstention is calibrated to causal
ambiguity rather than reserved for the one perfectly symmetric fixture.

Abstention is measured everywhere, as the share of episodes whose posterior is
too flat to call. A mechanism that cannot say "I do not know" is not usable by
anything that has to act on the answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    SELF_APPLICABILITY_MARGIN,
    SELF_CONTROLLABILITY_WEIGHT,
    TrackHistory,
    identify_roles,
    self_log_evidence,
    self_posterior,
)
from .successor_transfer import SlotReader, build_slot_reader

EXPERIMENT_ID = "brainworkshop-self-model-adversarial-2026-08-16"
SELF_MODEL_ADVERSARIAL_SCHEMA = "neural-computer.self-model-adversarial.v2"
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
MIMIC_VARIANTS = (
    "delayed_mimic",
    "partial_mimic",
    "stochastic_mimic",
    "distractor_controller",
)


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


def uniform_posterior(histories) -> list[float]:
    """An explicit abstention arm, useful as the episodic null baseline."""

    width = len(histories)
    return [1.0 / width] * width if width else []


def likelihood_gated_posterior(
    histories, counts, *, alphabet: int
) -> list[float]:
    """Applicability-gated likelihood, without the controllability term."""

    return self_posterior(
        histories,
        counts,
        alphabet=alphabet,
        controllability_weight=0.0,
    )


def _histories_from_trace(
    trace: list[tuple[int, ...]], actions: list[int]
) -> list[TrackHistory]:
    """Build scoring-side histories from a transformed correspondence trace."""

    width = max((len(row) for row in trace), default=0)
    histories = [TrackHistory() for _ in range(width)]
    for step in range(max(0, len(trace) - 1)):
        for track in range(min(len(trace[step]), len(trace[step + 1]))):
            histories[track].observe(
                trace[step][track], actions[step], trace[step + 1][track]
            )
    return histories


def _near_mimic(
    collected: list[dict[str, Any]],
    *,
    alphabet: int,
    variant: str,
) -> list[dict[str, Any]]:
    """Make a causal near-mimic from an already-paid observation stream.

    These are scoring-side controls. They never call the verifier and therefore
    cannot increase unique experience. Track zero remains the true agent; the
    transformed second track is the distractor under test.
    """

    if variant not in MIMIC_VARIANTS:
        raise ValueError(f"unknown mimic variant: {variant}")
    transformed = copy.deepcopy(collected)
    for episode_index, episode in enumerate(transformed):
        original = [tuple(row) for row in episode["trace"]]
        if not original or not original[0]:
            continue
        rows: list[tuple[int, ...]] = []
        previous = int(original[0][0])
        for step, row in enumerate(original):
            agent = int(row[0])
            action = int(episode["actions"][min(step, len(episode["actions"]) - 1)])
            if variant == "delayed_mimic":
                distractor = previous
            elif variant == "partial_mimic":
                # Respond on a fixed, known-to-the-audit half of actions and
                # otherwise hold. The branch is deterministic, not tuned to
                # the eventual score.
                distractor = agent if action % 2 == 0 else previous
            elif variant == "stochastic_mimic":
                # A deterministic pseudo-random coin keeps the record exactly
                # reproducible without paying another interaction stream.
                coin = (episode_index * 17 + step * 13 + action * 7) % 10
                distractor = agent if coin < 7 else previous
            else:  # distractor_controller
                distractor = (agent * 3 + action * 5 + 1) % max(2, alphabet)
            rows.append((agent, distractor, *row[2:]))
            previous = agent
        episode["trace"] = rows
        episode["histories"] = _histories_from_trace(rows, episode["actions"])
    return transformed


def _model_applicable(
    histories: list[TrackHistory], counts, *, alphabet: int
) -> tuple[bool, float, float]:
    """Return whether a remembered model still explains any current track."""

    if not histories:
        return False, float("-inf"), 0.0
    evidence = [
        self_log_evidence(history, counts, alphabet=alphabet)
        for history in histories
    ]
    steps = max(1, max(len(history.steps) for history in histories))
    uniform = -math.log2(1.0 / max(2, int(alphabet)))
    margin = (max(evidence) / steps) + uniform
    return margin >= SELF_APPLICABILITY_MARGIN, margin, max(evidence)


def causal_reversal_stream(
    pre_change: list[dict[str, Any]],
    post_change: list[dict[str, Any]],
    *,
    reader: SlotReader,
    pre_task: NavigationTask,
    post_task: NavigationTask,
    pre_model=None,
    accounting: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Test invalidation and recovery one post-change episode at a time.

    The pre-change model is frozen until its predictive applicability falls
    below the same margin used by ``self_posterior``. At that point it is
    quarantined: the current episode is forced to abstain, and subsequent
    episodes are used to rebuild a fresh model from only post-change evidence.
    No oracle labels participate in the decision or rebuild.
    """

    if pre_model is None:
        _, model = refit(
            pre_change,
            initial_posteriors(pre_change),
            reader=reader,
            task=pre_task,
            passes=PASSES,
            accounting=accounting,
        )
    else:
        model = pre_model
    recovery_buffer: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    quarantined = False
    detection_episode: int | None = None
    recovery_episode: int | None = None
    detection_events: list[int] = []
    recovery_events: list[int] = []
    consecutive_recovered = 0
    for index, episode in enumerate(post_change):
        applicable, margin, best_evidence = _model_applicable(
            episode["histories"], model, alphabet=reader.alphabet
        )
        if not quarantined and not applicable:
            quarantined = True
            detection_events.append(index)
            if detection_episode is None:
                detection_episode = index
            recovery_buffer = []
            consecutive_recovered = 0

        if quarantined:
            # The episode that exposed the collapse is not used to make a
            # confident call. It is retained as untrusted data for recovery.
            posterior = uniform_posterior(episode["histories"])
            recovery_buffer.append(episode)
            if len(recovery_buffer) >= 2:
                _, candidate = refit(
                    recovery_buffer,
                    initial_posteriors(recovery_buffer),
                    reader=reader,
                    task=post_task,
                    passes=1,
                    accounting=accounting,
                )
                candidate_posteriors = [
                    self_posterior(
                        item["histories"], candidate, alphabet=reader.alphabet
                    )
                    for item in recovery_buffer[-2:]
                ]
                candidate_scores = [
                    score([item], [candidate_posterior])
                    for item, candidate_posterior in zip(
                        recovery_buffer[-2:], candidate_posteriors
                    )
                ]
                if all(
                    item["named"] == 1.0 and item["confidently_wrong"] == 0.0
                    for item in candidate_scores
                ):
                    consecutive_recovered += 1
                else:
                    consecutive_recovered = 0
                if consecutive_recovered >= 1 and recovery_episode is None:
                    recovery_episode = index
                    model = candidate
                    quarantined = False
                    recovery_events.append(index)
                    recovery_buffer = []
                elif consecutive_recovered >= 1:
                    model = candidate
                    quarantined = False
                    recovery_events.append(index)
                    recovery_buffer = []
        else:
            posterior = self_posterior(
                episode["histories"], model, alphabet=reader.alphabet
            )

        episode_score = score([episode], [posterior])
        rows.append(
            {
                "episode": index,
                "applicable_before_call": applicable,
                "applicability_margin": margin,
                "best_log_evidence": best_evidence,
                "quarantined": quarantined,
                "posterior": posterior,
                **episode_score,
            }
        )

    named = sum(row["named"] for row in rows) / max(1, len(rows))
    return {
        "summary": {
            "named": named,
            "abstained": 1.0 - named,
            "precision": (
                sum(
                    row["named"] - row["confidently_wrong"]
                    for row in rows
                )
                / max(1.0, sum(row["named"] for row in rows))
            ),
            "confidently_wrong": sum(
                row["confidently_wrong"] for row in rows
            )
            / max(1, len(rows)),
        },
        "detected": detection_episode is not None,
        "detection_episode": detection_episode,
        "detection_events": detection_events,
        "recovered": recovery_episode is not None,
        "recovery_episode": recovery_episode,
        "recovery_events": recovery_events,
        "rows": rows,
    }


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


def iterative_recovery(
    collected,
    posteriors,
    *,
    reader: SlotReader,
    task: NavigationTask,
    passes: int = PASSES,
    accounting: dict[str, int] | None = None,
) -> list[dict[str, float]]:
    """Record every guarded pass after a deliberately poisoned start."""

    trace: list[dict[str, float]] = []
    current = posteriors
    for _ in range(int(passes) + 1):
        trace.append(score(collected, current))
        if accounting is not None:
            accounting["replayed_episode_histories"] += len(collected)
        model, _ = weighted_models(
            collected, current, reader=reader, task=task
        )
        current = [
            self_posterior(
                episode["histories"], model, alphabet=reader.alphabet
            )
            for episode in collected
        ]
    return trace


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

        # The ground moves without notification. This is a causal stream:
        # freeze the pre-change model, quarantine it only after predictive
        # collapse, and rebuild from post-change evidence one episode at a
        # time. There is no hindsight refit over both worlds.
        reversal = causal_reversal_stream(
            collected,
            elsewhere,
            reader=reader,
            pre_task=task,
            post_task=other,
            pre_model=self_counts,
            accounting=accounting,
        )
        entry["reversed"] = reversal["summary"]
        entry["reversal_stream"] = reversal

        # Deliberately started on the wrong track.
        poisoned, _ = refit(
            collected,
            poisoned_posteriors(collected),
            reader=reader,
            task=task,
            accounting=accounting,
        )
        entry["poisoned"] = score(collected, poisoned)
        entry["poisoned_recovery"] = iterative_recovery(
            collected,
            poisoned_posteriors(collected),
            reader=reader,
            task=task,
            accounting=accounting,
        )

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

        near_mimics: dict[str, dict[str, float]] = {}
        for variant in MIMIC_VARIANTS:
            transformed = _near_mimic(
                collected, alphabet=reader.alphabet, variant=variant
            )
            transformed_posteriors, _ = refit(
                transformed,
                initial_posteriors(transformed),
                reader=reader,
                task=task,
                accounting=accounting,
            )
            near_mimics[variant] = score(transformed, transformed_posteriors)
        entry["near_mimics"] = near_mimics

        # The guard and controllability term were added after the first
        # integrated result. Keep the exact earlier posterior beside the new
        # one so an apparent rescue cannot be attributed to an unreported
        # change of baseline.
        episodic_honest = initial_posteriors(collected)
        episodic_poisoned = poisoned_posteriors(collected)
        old_honest, _ = refit(
            collected,
            episodic_honest,
            reader=reader,
            task=task,
            posterior_fn=likelihood_only_posterior,
            accounting=accounting,
        )
        old_poisoned, _ = refit(
            collected,
            episodic_poisoned,
            reader=reader,
            task=task,
            posterior_fn=likelihood_only_posterior,
            accounting=accounting,
        )
        gated_honest, _ = refit(
            collected,
            episodic_honest,
            reader=reader,
            task=task,
            posterior_fn=likelihood_gated_posterior,
            accounting=accounting,
        )
        gated_poisoned, _ = refit(
            collected,
            episodic_poisoned,
            reader=reader,
            task=task,
            posterior_fn=likelihood_gated_posterior,
            accounting=accounting,
        )
        entry["ablation"] = {
            "guarded_honest": entry["honest"],
            "guarded_poisoned": entry["poisoned"],
            "likelihood_only_honest": score(collected, old_honest),
            "likelihood_only_poisoned": score(collected, old_poisoned),
        }
        entry["baseline_arms"] = {
            "episodic_identity": {
                "honest": score(collected, episodic_honest),
                "poisoned": score(collected, episodic_poisoned),
            },
            "remembered_likelihood_only": {
                "honest": score(collected, old_honest),
                "poisoned": score(collected, old_poisoned),
            },
            "remembered_likelihood_gated": {
                "honest": score(collected, gated_honest),
                "poisoned": score(collected, gated_poisoned),
            },
            "remembered_likelihood_controllable": {
                "honest": entry["honest"],
                "poisoned": entry["poisoned"],
            },
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
    near_mimic_summary = {
        variant: {
            key: sum(row["near_mimics"][variant][key] for row in rows) / len(rows)
            for key in ("named", "abstained", "precision", "confidently_wrong")
        }
        for variant in MIMIC_VARIANTS
    }
    baseline_names = tuple(rows[0]["baseline_arms"]) if rows else ()
    baseline_arms = {
        name: {
            polarity: {
                key: sum(
                    row["baseline_arms"][name][polarity][key] for row in rows
                )
                / len(rows)
                for key in ("named", "abstained", "precision", "confidently_wrong")
            }
            for polarity in ("honest", "poisoned")
        }
        for name in baseline_names
    }
    recovery_passes = [
        {
            key: sum(row["poisoned_recovery"][index][key] for row in rows)
            / len(rows)
            for key in ("named", "abstained", "precision", "confidently_wrong")
        }
        for index in range(
            max((len(row["poisoned_recovery"]) for row in rows), default=0)
        )
    ]
    report = {
        "schema": SELF_MODEL_ADVERSARIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": len(rows),
        "explore_episodes": explore_episodes,
        "confident_at": CONFIDENT,
        "mechanism": {
            "applicability_margin": SELF_APPLICABILITY_MARGIN,
            "controllability_weight": SELF_CONTROLLABILITY_WEIGHT,
            "frozen_for_holdout": True,
        },
        "conditions": summary,
        "ablations": ablations,
        "baseline_arms": baseline_arms,
        "poisoned_recovery": recovery_passes,
        "near_mimics": near_mimic_summary,
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
