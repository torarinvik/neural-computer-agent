"""One loop: observe, recognise, induce, execute, confirm, admit.

Every capability built in this session so far runs in experimenter Python and
ends at a table. Feedback inversion, noise-tolerant fitting, class escalation,
regime tracking and the accumulation curve have never met each other inside a
single agent, and nothing any of them produced was ever kept. This module is
that loop, and the thing it does that none of the others did is **keep what it
learns**.

The mechanism that makes accumulation pay is the *probe ladder*, and it is
worth being precise about why, because the obvious design does not work.

Recognition that runs after a fixed evidence budget saves nothing. The agent
still buys every episode, still reads every label, and only avoids the search
at the end -- which costs CPU, not experience, and the objective is written in
experience. So the budget is not fixed. The agent buys **one short episode**,
asks the library whether anything it already has explains it, and only buys
more when the answer is no. A task the library covers costs one episode. A
task it does not cover walks the ladder to the budget induction needs and pays
the full price, exactly as a fresh agent would.

That makes the two arms differ in the currency that matters:

- **growing** -- admitted programs persist, so a recurring task is recognised
  from one episode;
- **control** -- the library is restored before every task, so every task is
  met by an agent that has never seen anything.

The task stream has repeats, and that is not a convenience. An i.i.d. stream of
distinct rules is a distribution on which no library can pay, which the
composition record already established the hard way; a world worth carrying a
library into is one that revisits situations.

Nothing here reads `config.rule`, and nothing imports a symbol oracle. The
alphabet comes from clustering the frontend's own events, the target comes from
inverting the agent's own per-step reward, and the hypothesis is compiled by
provenance-neutral code that never learns where its machine came from.
`AgentBrain.bank` is not touched: the temporal family provably cannot express
most of these rules, so the store that grows here is the append-only induced
library, checksummed and persisted like any other brain artifact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from neural_computer import ExternalTemporalProgramBank
from neural_computer.induced_library import (
    INDUCED_LIBRARY_EXTENSION,
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
)
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config, curriculum_rules
from .compositional_recognition import composed_machine, search_compositions
from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import (
    compile_rule as compile_automaton,
)
from .counter_state_programs import (
    initial_counters,
    predict_symbols,
    run_counter_program,
)
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .identification_ceiling import Trace, episode_trace
from .lease_discrimination import DISCRIMINATION_ALPHA, binomial_upper_tail
from .noise_tolerant_induction import induce_noise_tolerant
from .prototype_templates import (
    cluster_events,
    estimated_tolerance,
    observe_events,
)
from .rule_automata import RuleAutomaton, positive_rate

EXPERIMENT_ID = "brainworkshop-integrated-agent-2026-08-15"
INTEGRATED_SCHEMA = "neural-computer.integrated-agent.v1"

# Already consumed by earlier development work; the holdout run names its own
# block in the seed ledger.
DEVELOPMENT_SEED = 41
# Episodes the agent probes with are short, because the identification record
# found that the same number of labels identifies far more rules when it
# arrives segmented than when it arrives as one long episode.
PROBE_STEPS = 16
CONFIRMATION_STEPS = 448
CONFIRMATION_EPISODES = 2
# Rungs the agent buys evidence in. It stops at the first rung where something
# explains what it has seen, so a covered task never reaches the last one.
PROBE_LADDER = (1, 2, 4, 8, 16, 28)
# Below this much evidence, fitting a machine from scratch is not attempted:
# the identification record measured that it needs roughly this many short
# episodes, and searching under it wastes the search rather than the evidence.
INDUCTION_LADDER_INDEX = 3
# Seeds one task apart. Probe episodes are drawn at `seed + 1000 + index`, so a
# narrow stride makes neighbouring tasks share stimulus streams outright -- they
# would still be scored by their own rules, but they would not be independent
# draws, and the record would say they were.
TASK_SEED_STRIDE = 10_000


@dataclass
class TaskOutcome:
    """What one task cost, how it was solved, and whether it was kept."""

    rule_digest: str
    state_count: int
    repeat_index: int
    probe_episodes: int = 0
    confirmation_episodes: int = 0
    source: str = "unsolved"
    library_slot: int | None = None
    library_size_before: int = 0
    fit_error_rate: float | None = None
    inferred_state_count: int | None = None
    confirmation_accuracies: list[float] = field(default_factory=list)
    pooled_accuracy: float | None = None
    reproduces: bool | None = None
    solved: bool = False
    admitted: bool = False
    admission_reason: str | None = None
    executor_statuses: str | None = None
    false_recognitions: int = 0
    composition_hypotheses: int = 0
    composed_from: list[int] = field(default_factory=list)
    combiner: str | None = None
    candidate_label: str | None = None

    @property
    def acquisition_steps(self) -> int:
        """Evidence spent choosing a hypothesis. What accumulation is about."""

        return self.probe_episodes * PROBE_STEPS

    @property
    def verification_steps(self) -> int:
        """Evidence spent proving one. A tax both arms pay per candidate."""

        return self.confirmation_episodes * CONFIRMATION_STEPS

    @property
    def verifier_steps(self) -> int:
        """The honest currency: labelled steps bought from the verifier."""

        return self.acquisition_steps + self.verification_steps

    def payload(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["acquisition_steps"] = self.acquisition_steps
        data["verification_steps"] = self.verification_steps
        data["verifier_steps"] = self.verifier_steps
        return data


def _record_for(
    machine: RuleAutomaton,
    *,
    alphabet: int,
    provenance: dict[str, Any],
) -> InducedProgramRecord:
    """Compile a hypothesis into a storable, self-describing program file.

    The signature is taken from the **compiled program**, not from the machine
    it came from. That makes it a certificate of the artifact actually being
    stored rather than of the hypothesis that motivated it, and it would catch
    a compiler that quietly changed behaviour.
    """

    program = compile_automaton(
        machine,
        channel_of_symbol=tuple(range(alphabet)),
        cluster_count=alphabet,
    )
    start = initial_counters(
        program, cluster_count=alphabet, states=machine.state_count
    )
    signature, statuses = predict_symbols(
        program,
        canonical_signature_stream(alphabet),
        cluster_count=alphabet,
        initial_counters=start,
    )
    if statuses != ("halted",):
        raise RuntimeError(f"compiled program did not halt cleanly: {statuses}")
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=int(alphabet),
        signature=signature,
        provenance=dict(provenance),
    ).validate()


def _agreement(record: InducedProgramRecord, traces) -> tuple[int, int]:
    """Hits and scored steps for one stored program against evidence in hand.

    Reads no reward and runs no episode: the labels were already bought, and
    asking a program what it would have pressed is arithmetic.
    """

    hits = trials = 0
    for trace in traces:
        presses, _ = predict_symbols(
            record.program,
            trace.symbols,
            cluster_count=record.alphabet,
            initial_counters=record.initial_counters,
        )
        for position, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            hits += int(presses[position] == trace.outputs[position])
    return hits, trials


def proves_competence(
    hits: int,
    trials: int,
    *,
    threshold: float = THRESHOLD,
    alpha: float = DISCRIMINATION_ALPHA,
) -> bool:
    """Is this too good for a policy that merely sits at the gate?

    The direction of this test is the whole point, and getting it backwards
    is what a first version of this module did. `control_below_threshold_report`
    asks whether an arm can be *ruled out* as competent, and over sixteen
    labels almost nothing can be ruled out either way -- so a wrong program
    passed, was adopted, and reproduced at 0.73 on held-out episodes.

    Adoption needs the opposite: evidence *for* competence. A policy sitting
    exactly at the gate would produce a run this good with probability at most
    `alpha`, or the evidence is not yet enough and the agent buys another rung.
    Over sixteen labels even a perfect program cannot clear this, which is
    correct -- sixteen labels do not distinguish a perfect program from a
    slightly wrong one -- and over thirty-two it can.
    """

    if trials < 1:
        return False
    observed = hits / trials
    if observed < threshold:
        return False
    return binomial_upper_tail(trials, threshold, observed) <= alpha


def recognise(
    library: InducedProgramLibrary,
    traces,
    *,
    threshold: float = THRESHOLD,
    exclude: frozenset[int] = frozenset(),
) -> tuple[int, float] | None:
    """The best stored program the evidence positively supports, if any."""

    best: tuple[int, float] | None = None
    for slot in range(library.record_count):
        if slot in exclude:
            continue
        hits, trials = _agreement(library.record(slot), traces)
        if not proves_competence(hits, trials, threshold=threshold):
            continue
        rate = hits / trials
        if best is None or rate > best[1]:
            best = (slot, rate)
    return best


def _probe(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    config,
    clusters: torch.Tensor,
    *,
    seed: int,
    index: int,
) -> Trace:
    """Buy one short scored episode and read it as a labelled trace."""

    return episode_trace(
        payload, encoders, bank, config, clusters, seed=seed + 1000 + index
    )


def discover_alphabet(encoders, config, *, seed: int) -> torch.Tensor:
    """Find the letters once, and never again.

    `cluster_events` is greedy and first-come: a cluster's index is the order
    it first appeared in the stream it was discovered on. Rediscovering the
    alphabet per task therefore gives every task a private and mutually
    incomparable set of symbol names, and a program induced under one of them
    means nothing under another -- which would have made every recognition in
    this module a coincidence.

    So the agent establishes one alphabet, from one observation pass, and every
    later task speaks it. This costs no reward: the stimulus stream is drawn
    independently of the rule, and clustering reads only the frontend's own
    events.

    The tolerance is measured rather than assumed. A fixed 0.5 is correct only
    in a room where the same symbol renders identically every time; under pixel
    noise the within-stimulus spread passes it and every symbol shatters. When
    the stimuli genuinely overlap the estimator returns nothing, and this
    raises rather than clustering at a number it made up.
    """

    events = observe_events(encoders, config, seed=seed)
    tolerance = estimated_tolerance(events)
    if tolerance is None:
        raise ValueError("the stimuli do not separate into an alphabet")
    # The cap is deliberately far above any alphabet being tried. At the
    # default of eight, an eight-symbol room whose stimuli had actually
    # shattered still reported eight letters, and the saturated count read as
    # success.
    return cluster_events(events, tolerance=tolerance, maximum_clusters=32)


def solve_task(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    library: InducedProgramLibrary,
    rule,
    clusters: torch.Tensor,
    *,
    seed: int,
    repeat_index: int,
    ladder: tuple[int, ...] = PROBE_LADDER,
    induction_from: int = INDUCTION_LADDER_INDEX,
    corrupt=None,
    config_for=_config,
    compose: bool = False,
    correct_for_multiplicity: bool = False,
) -> tuple[TaskOutcome, InducedProgramRecord | None]:
    """Walk the ladder until something explains the task, then confirm it.

    `config_for` is how the environment is varied without varying the agent.
    The widening sweep passes one that renders a larger alphabet or adds pixel
    noise; nothing in the loop below knows which.
    """

    outcome = TaskOutcome(
        rule_digest=rule.digest(),
        state_count=rule.state_count,
        repeat_index=repeat_index,
        library_size_before=library.record_count,
    )
    probe_config = config_for(rule, PROBE_STEPS)
    alphabet = int(clusters.shape[0])

    full = config_for(rule, CONFIRMATION_STEPS)

    def confirm(record: InducedProgramRecord, attempt: int) -> tuple[float, str]:
        """Run a candidate in the environment on episodes it never saw.

        This is the only place the executor meets the verifier, and every
        episode is counted -- including the ones spent on candidates that turn
        out to be wrong. An agent that only paid for its successes would be
        measuring something other than what it costs to be this agent.
        """

        accuracies: list[float] = []
        statuses: set[str] = set()
        for offset in range(CONFIRMATION_EPISODES):
            executed = run_counter_program(
                record.program,
                encoders,
                full,
                clusters,
                seed=seed + 100 + 10 * attempt + offset,
                initial_counters=record.initial_counters,
            )
            accuracies.append(float(executed["accuracy"]))
            statuses.add(str(executed["statuses"]))
        outcome.confirmation_episodes += CONFIRMATION_EPISODES
        outcome.confirmation_accuracies.extend(accuracies)
        outcome.executor_statuses = ",".join(sorted(statuses))
        return sum(accuracies) / len(accuracies), ",".join(sorted(statuses))

    traces: list[Trace] = []
    # A recognition that fails confirmation is information, not a dead end. The
    # slot is refused for this task and the ladder resumes, which is what makes
    # a false recognition cost evidence rather than cost the task.
    refused_slots: set[int] = set()
    # A composed candidate is not a slot, so refusals are tracked by the label
    # of the hypothesis rather than by where it came from.
    refused_labels: set[str] = set()
    attempt = 0
    for rung, budget in enumerate(ladder):
        while len(traces) < budget:
            trace = _probe(
                payload,
                encoders,
                bank,
                probe_config,
                clusters,
                seed=seed,
                index=len(traces),
            )
            # `corrupt` exists for the missing-evidence controls. An agent whose
            # feedback has been destroyed must fail to learn; one that still
            # reports solved tasks is reading something other than its reward.
            traces.append(trace if corrupt is None else corrupt(trace, len(traces)))
        outcome.probe_episodes = len(traces)

        candidate: InducedProgramRecord | None = None
        source = ""
        slot: int | None = None
        found = None
        if compose:
            # The library is asked what it can *build*, not only what it holds.
            # Every pair of records under every combiner is one elementwise
            # merge of two cached press vectors, so this costs no episode and
            # no program execution -- but it multiplies the hypotheses, and the
            # threshold rises with their number to match.
            offer, search = search_compositions(
                library,
                traces,
                exclude=frozenset(refused_slots),
                correct_for_multiplicity=correct_for_multiplicity,
            )
            outcome.composition_hypotheses = int(search["hypotheses"])
            if offer is not None and offer.label() not in refused_labels:
                built = composed_machine(library, offer)
                if built is not None:
                    if offer.kind == "single":
                        slot = offer.slots[0]
                        candidate = library.record(slot)
                        source = "recognised"
                    else:
                        candidate = _record_for(
                            built,
                            alphabet=alphabet,
                            provenance={
                                "source": "composed",
                                "parts": list(offer.slots),
                                "combiner": offer.combiner,
                                "probe_episodes": len(traces),
                                "hypotheses_examined": int(search["hypotheses"]),
                                "states": built.state_count,
                                "machine": built.payload(),
                            },
                        )
                        source = "composed"
                        outcome.composed_from = list(offer.slots)
                        outcome.combiner = offer.combiner
                    outcome.candidate_label = offer.label()
        else:
            found = recognise(library, traces, exclude=frozenset(refused_slots))
            if found is not None:
                slot, _ = found
                candidate = library.record(slot)
                source = "recognised"
                outcome.candidate_label = f"slot {slot}"
        if candidate is None and rung >= induction_from:
            fit = induce_noise_tolerant(tuple(traces))
            if fit is not None:
                outcome.fit_error_rate = fit.error_rate
                outcome.inferred_state_count = fit.machine.state_count
                # A fit whose own disagreement rate already leaves it under the
                # gate is not worth confirming: another rung of evidence is
                # cheaper than two full episodes spent on a hypothesis that
                # says it is wrong.
                if 1.0 - fit.error_rate >= THRESHOLD or budget == ladder[-1]:
                    candidate = _record_for(
                        fit.machine,
                        alphabet=alphabet,
                        provenance={
                            "source": "induced",
                            "probe_episodes": len(traces),
                            "probe_steps": PROBE_STEPS,
                            "fit_error_rate": fit.error_rate,
                            "states": fit.machine.state_count,
                            "machine": fit.machine.payload(),
                        },
                    )
                    source = "induced"
        if candidate is None:
            continue

        pooled, _ = confirm(candidate, attempt)
        attempt += 1
        trials = CONFIRMATION_EPISODES * CONFIRMATION_STEPS
        # The same test, in the same direction, as recognition. A first version
        # asked only whether the candidate could be *ruled out* as competent,
        # and a program fitted to shuffled feedback cleared it at 0.814 against
        # a gate of 0.8 -- which is precisely the near-miss the lease machinery
        # exists to refuse. Confirmation has to prove competence too.
        reproduces = proves_competence(
            round(pooled * trials), trials, threshold=THRESHOLD
        )
        outcome.pooled_accuracy = pooled
        if reproduces:
            outcome.source = source
            outcome.library_slot = slot
            outcome.reproduces = True
            outcome.solved = True
            return outcome, candidate
        outcome.reproduces = False
        if source in ("recognised", "composed"):
            if slot is not None:
                refused_slots.add(slot)
            if outcome.candidate_label:
                refused_labels.add(outcome.candidate_label)
            outcome.false_recognitions += 1
        else:
            # An induced hypothesis that fails confirmation is refuted by more
            # evidence than it was built from; more probes is the only move.
            continue

    outcome.admission_reason = "nothing explained the evidence within the ladder"
    return outcome, None


def admit(
    library: InducedProgramLibrary,
    record: InducedProgramRecord,
    outcome: TaskOutcome,
    *,
    library_path: Path | None = None,
) -> None:
    """Keep a confirmed program, unless the library already presses that way.

    Admission-by-compression in the only currency this family has. A duplicate
    is refused on the signature index alone -- no execution, no evidence -- so
    a growing library does not fill up with restatements of what it already
    knows.
    """

    if outcome.source == "recognised":
        outcome.admission_reason = "already in the library"
        return
    if not outcome.reproduces:
        outcome.admission_reason = "did not reproduce on held-out episodes"
        return
    duplicate = library.duplicate_of(record.signature)
    if duplicate is not None:
        outcome.admission_reason = f"presses identically to slot {duplicate}"
        outcome.library_slot = duplicate
        return
    slot = library.append(record)
    outcome.admitted = True
    outcome.library_slot = slot
    outcome.admission_reason = "admitted"
    if library_path is not None:
        library.save(library_path)


def noisy_feedback(rate: float, seed: int):
    """Flip a fraction of each probe's labels, rather than destroying them.

    Shuffling is the missing-evidence control and answers whether the agent is
    reading its reward at all. This is the different and harder question: does
    composition survive a verifier that is merely *unreliable*? The noise
    tolerance record establishes that induction does, up to one label in five;
    composition has to be asked separately, because it is scored by a test that
    does not move when the evidence gets dirtier.
    """

    if not 0.0 <= rate < 1.0:
        raise ValueError("a feedback noise rate is a fraction below one")

    def corrupt(trace: Trace, index: int) -> Trace:
        generator = torch.Generator().manual_seed(int(seed) + 7919 * int(index))
        flips = (
            torch.rand(len(trace.outputs), generator=generator) < rate
        ).tolist()
        return Trace(
            symbols=trace.symbols,
            outputs=tuple(
                value ^ int(flip)
                for value, flip in zip(trace.outputs, flips, strict=True)
            ),
            eligible=trace.eligible,
            symbol_count=trace.symbol_count,
        )

    return corrupt


def shuffled_feedback(seed: int):
    """The reward-shuffled control, as a trace corruption.

    Permuting a trace's labels leaves the label marginal exactly where it was
    and destroys only the relation between symbol and press -- which is the
    single thing the whole pipeline claims to recover. An agent that keeps
    solving tasks under this is reading its rewards' *frequency* rather than
    their *content*, and every number in this record would mean nothing.
    """

    def corrupt(trace: Trace, index: int) -> Trace:
        generator = torch.Generator().manual_seed(int(seed) + int(index))
        order = torch.randperm(len(trace.outputs), generator=generator).tolist()
        return Trace(
            symbols=trace.symbols,
            outputs=tuple(trace.outputs[position] for position in order),
            eligible=trace.eligible,
            symbol_count=trace.symbol_count,
        )

    return corrupt


def task_stream(
    rules,
    *,
    length: int,
    pool_size: int,
    seed: int,
) -> tuple[tuple[int, Any, int], ...]:
    """A stream of tasks drawn with repetition from a small pool.

    Repetition is the whole experiment. On a stream of all-distinct tasks no
    library can pay, and reporting that as a null result would say nothing
    about libraries -- only about the stream.
    """

    if pool_size < 1 or pool_size > len(rules):
        raise ValueError("pool size must fit inside the rule population")
    if length < 1:
        raise ValueError("a task stream needs at least one task")
    # Spread the pool across the complexity axis rather than taking a prefix.
    # `curriculum_rules` is ordered by state count, so a prefix is a pool of
    # nothing but the easiest rules, and a library that pays only on those
    # would be a result about one-state machines.
    stride = max(1, len(rules) // pool_size)
    pool = list(rules)[:: stride][:pool_size]
    if len(pool) < pool_size:
        pool = list(rules)[:pool_size]
    generator = torch.Generator().manual_seed(int(seed))
    draws = torch.randint(0, pool_size, (int(length),), generator=generator).tolist()
    seen: dict[int, int] = {}
    stream = []
    for position, choice in enumerate(draws):
        repeat = seen.get(choice, 0)
        seen[choice] = repeat + 1
        stream.append((position, pool[choice], repeat))
    return tuple(stream)


def run_arm(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    stream,
    clusters: torch.Tensor,
    *,
    grow: bool,
    seed: int,
    frontend_digest: str,
    library_path: Path | None = None,
    corrupt=None,
    config_for=_config,
    compose: bool = False,
) -> dict[str, Any]:
    """One pass over the task stream. `grow=False` forgets between tasks."""

    alphabet = int(clusters.shape[0])
    library = InducedProgramLibrary(
        alphabet=alphabet, frontend_digest=frontend_digest
    )
    rows: list[dict[str, Any]] = []
    for position, rule, repeat in stream:
        if not grow:
            library = InducedProgramLibrary(
                alphabet=alphabet, frontend_digest=frontend_digest
            )
        outcome, record = solve_task(
            payload,
            encoders,
            bank,
            library,
            rule,
            clusters,
            seed=seed + TASK_SEED_STRIDE * position,
            repeat_index=repeat,
            corrupt=corrupt,
            config_for=config_for,
            compose=compose,
        )
        if record is not None:
            admit(
                library,
                record,
                outcome,
                library_path=library_path if grow else None,
            )
        row = outcome.payload()
        # Verifier-side annotations, for reading the table only. The agent
        # never sees these, and nothing in the loop consults them.
        rate = positive_rate(rule, seed=seed + TASK_SEED_STRIDE * position)
        row["positive_rate"] = rate
        # A task a constant policy already clears proves nothing about
        # induction, recognition, or libraries. Counting those separately is
        # what keeps the headline number about the mechanism.
        row["trivial"] = max(rate, 1.0 - rate) >= THRESHOLD
        rows.append(row)
    hard = [row for row in rows if not row["trivial"]]
    return {
        "grew_the_library": grow,
        "tasks": len(rows),
        "trivial_tasks": len(rows) - len(hard),
        "solved": sum(1 for row in rows if row["solved"]),
        "solved_nontrivial": sum(1 for row in hard if row["solved"]),
        "nontrivial_tasks": len(hard),
        "nontrivial_acquisition_steps": sum(
            int(row["acquisition_steps"]) for row in hard
        ),
        "recognised": sum(1 for row in rows if row["source"] == "recognised"),
        "composed": sum(1 for row in rows if row["source"] == "composed"),
        "induced": sum(1 for row in rows if row["source"] == "induced"),
        "admitted": sum(1 for row in rows if row["admitted"]),
        "final_library_size": library.record_count,
        "false_recognitions": sum(int(row["false_recognitions"]) for row in rows),
        "total_probe_episodes": sum(int(row["probe_episodes"]) for row in rows),
        "total_acquisition_steps": sum(int(row["acquisition_steps"]) for row in rows),
        "total_verification_steps": sum(int(row["verification_steps"]) for row in rows),
        "total_verifier_steps": sum(int(row["verifier_steps"]) for row in rows),
        "rows": rows,
    }


def run_integrated(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    stream_length: int = 12,
    pool_size: int = 4,
    library_path: Path | None = None,
) -> dict[str, Any]:
    """Both arms over one matched task stream, and what the growing arm kept."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    rules = curriculum_rules()
    stream = task_stream(
        rules, length=stream_length, pool_size=pool_size, seed=seed
    )
    # One alphabet, established once from a full-length observation pass, and
    # spoken by every task and every stored program thereafter.
    clusters = discover_alphabet(
        encoders, _config(rules[0], CONFIRMATION_STEPS), seed=seed
    )
    alphabet = int(clusters.shape[0])

    started = time.perf_counter()
    growing = run_arm(
        payload,
        encoders,
        bank,
        stream,
        clusters,
        grow=True,
        seed=seed,
        frontend_digest=encoders.digest(),
        library_path=library_path,
    )
    control = run_arm(
        payload,
        encoders,
        bank,
        stream,
        clusters,
        grow=False,
        seed=seed,
        frontend_digest=encoders.digest(),
    )
    shuffled = run_arm(
        payload,
        encoders,
        bank,
        stream,
        clusters,
        grow=True,
        seed=seed,
        frontend_digest=encoders.digest(),
        corrupt=shuffled_feedback(seed),
    )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the integrated agent mutated AgentBrain.bank")

    def _ratio(key: str) -> float | None:
        return growing[key] / control[key] if control[key] else None

    report = {
        "schema": INTEGRATED_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "alphabet": alphabet,
        "stream_length": stream_length,
        "pool_size": pool_size,
        "distinct_tasks": len({rule.digest() for _, rule, _ in stream}),
        "growing": growing,
        "control": control,
        "reward_shuffled": shuffled,
        "acquisition_ratio": _ratio("total_acquisition_steps"),
        "nontrivial_acquisition_ratio": _ratio("nontrivial_acquisition_steps"),
        "verifier_step_ratio": _ratio("total_verifier_steps"),
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "integrated_agent.json").write_text(
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
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository / "session_records" / "brainworkshop_integrated_agent_2026-08-15"
        ),
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--stream-length", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--library", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.library is not None and (
        arguments.library.suffix != INDUCED_LIBRARY_EXTENSION
    ):
        parser.error(f"library path must end in {INDUCED_LIBRARY_EXTENSION}")
    report = run_integrated(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        stream_length=arguments.stream_length,
        pool_size=arguments.pool_size,
        library_path=arguments.library,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
