"""Does capability N+1 get cheaper as the library grows?

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` names this curve as its own
falsifier, and until now it did not exist. Every result in the repository so
far measures a *ceiling* -- what some family can represent -- and none of them
measures the quantity the objective is actually written in: verified reusable
capability per unique experience.

The measurement is a curriculum, not a sweep. Rules arrive one at a time. The
searcher runs against whatever library exists at that moment; when it gates a
winner, the winner is admitted and the library grows. Cost is the verifier
evidence spent before a rule gates. Two arms differ in one bit:

- **growing** -- winners are admitted, so rule N faces a library of everything
  learned from rules 1..N-1;
- **control** -- the library is restored to its founding state before every
  rule, so each rule is learned by a fresh agent.

If the architecture's story is true, growing costs less than control, and the
gap widens with library size. If the two arms lie on top of each other, the
external-program story is decoration and the capability lives in the network.

Both arms are diagnostics on an already-consumed development seed. Neither
touches `AgentBrain.bank`: the curriculum grows a scratch copy, and the real
bank is checksummed before and after.

Two honest caveats belong in the reading of any result here.

First, the searcher enumerates in a fixed order -- retrieve, compose, invert,
and, invent, templates -- and library-derived proposals come first. A larger
library therefore *lengthens* the prefix a rule must walk through before
reaching the templates that actually solve it. Accumulation has to beat that
headwind, and the per-rule breakdown records how much of the cost was spent on
the prefix.

Second, storability is not guaranteed. A winner the searcher can find is not
automatically a file the bank can hold; `_winner_artifact` returns `None` when
this family has no representation for the winning combination, and those rules
are counted rather than quietly dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.program import ExternalProgramArtifact
from neural_computer.promotion import sha256_file

from .bank_program import (
    admit_and_child,
    admit_temporal_program,
    and_artifact,
    invert_artifact,
    prototype_match_artifact,
)
from .behaviour_signature import (
    behaviour_signature,
    observe_stream,
    partition_by_behaviour,
)
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import (
    FRONTEND_SEED,
    MINIMUM_BITS,
    THRESHOLD,
    _machine,
    curated_frontend,
)
from .feedback_proposer import (
    probe_target,
    rank_by_agreement,
    signatures_for,
)
from .lease_discrimination import control_below_threshold_report
from .program_search import ProgramProposal, install_proposal, propose_from_bank
from .prototype_templates import observe_events, observed_templates
from .rendered_environment import RenderedBrainWorkshopConfig
from .rendered_live import run_rendered_live_lifetime
from .rule_automata import positive_rate, sample_rule

EXPERIMENT_ID = "brainworkshop-accumulation-curve-2026-08-15"
ACCUMULATION_SCHEMA = "neural-computer.accumulation-curve.v1"

# Already consumed by earlier development work; nothing fresh is spent here.
DEVELOPMENT_SEED = 41
STATE_COUNTS = (1, 2, 3, 4, 5, 6)
RULES_PER_STATE_COUNT = 3
SYMBOL_COUNT = 4
STEPS = 448
# Admission asks a winner to reproduce on episodes it did not gate on. These
# cost verifier evidence like any other episode and are counted as such.
CONFIRMATION_EPISODES = 2


def curriculum_rules(
    *,
    state_counts: tuple[int, ...] = STATE_COUNTS,
    rules_per_state_count: int = RULES_PER_STATE_COUNT,
    symbol_count: int = SYMBOL_COUNT,
) -> tuple:
    """The baseline's rules, in ascending complexity order.

    Deliberately the same population the baseline and the two ceilings were
    measured on, so the curve can be read against them without a new sample.
    """

    rules = []
    for states in state_counts:
        for index in range(rules_per_state_count):
            rule = sample_rule(
                symbol_count=symbol_count,
                state_count=states,
                seed=6000 + 100 * states + index,
            )
            if rule is not None:
                rules.append(rule)
    return tuple(rules)


def _config(rule, steps: int) -> RenderedBrainWorkshopConfig:
    return RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        symbol_count=rule.symbol_count,
        match_rule="automaton",
        rule=rule,
    )


def _winner_artifact(
    proposal: ProgramProposal,
    machine,
    bank: ExternalTemporalProgramBank,
    *,
    frontend_digest: str,
) -> tuple[ExternalProgramArtifact | None, int | None, str]:
    """The bank file a winning proposal becomes, if this family has one.

    Returns the artifact, the delay parent slot an AND child needs, and a
    reason. Polarity is the subtle case: `install_proposal` flips the
    intention on the machine rather than in the artifact, so a winning
    inverted template must be wrapped in an invert file or the bank would
    store a program that behaves differently when retrieved.
    """

    if proposal.kind == "retrieve":
        return None, None, "already in the library"
    if proposal.kind == "compose":
        return proposal.artifact, None, "same-primitive compose"
    if proposal.kind == "invert":
        return proposal.artifact, None, "invert child"
    if proposal.kind == "invent":
        prototype = machine.prototype.detach().cpu().reshape(1, -1)
        artifact = prototype_match_artifact(
            bank.context_width,
            prototype=prototype,
            frontend_digest=frontend_digest,
        )
        if proposal.invert_intention:
            return invert_artifact(artifact), None, "inverted prototype child"
        return artifact, None, "prototype child"
    if proposal.kind == "and":
        prototype = machine.prototype.detach().cpu().reshape(1, -1)
        artifact = and_artifact(prototype, frontend_digest=frontend_digest)
        return artifact, int(proposal.slots[0]), "and child"
    return None, None, f"no bank representation for {proposal.kind}"


def _find_proposal(
    proposals: tuple[ProgramProposal, ...], label: str
) -> ProgramProposal | None:
    for proposal in proposals:
        if proposal.label() == label:
            return proposal
    return None


def _library_prefix_length(proposals: tuple[ProgramProposal, ...]) -> int:
    """How many proposals precede the first freshly invented one.

    This is the headwind a growing library imposes on a rule it cannot help
    with: every one of these must be tried and rejected first.
    """

    for index, proposal in enumerate(proposals):
        if proposal.kind == "invent":
            return index
    return len(proposals)


def _learn_one_rule(
    payload: dict[str, object],
    bank_path: Path,
    encoders,
    rule,
    *,
    seed: int,
    steps: int,
    founding_program_count: int,
    proposer: str = "enumerate",
    compression_admission: bool = False,
) -> dict[str, Any]:
    """Search against the current library; return cost, winner, and file.

    `proposer` selects how candidates are ordered and filtered:

    - `enumerate` is the original fixed-order walk, every executable proposal
      paid for in turn. This is the arm the first curve measured.
    - `dedup` keeps that order but skips any proposal that presses identically
      to an earlier one on the episode it would be scored on. Lossless.
    - `feedback` spends one episode recovering the target behaviour from its
      own per-step rewards, ranks every candidate offline by agreement, and
      pays only for the ones worth confirming.
    """

    from .program_search import search_temporal_programs

    if proposer not in ("enumerate", "dedup", "feedback"):
        raise ValueError("proposer must be enumerate, dedup, or feedback")
    config = _config(rule, steps)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    machine = _machine(payload, learn=False)

    def acquire(proposal):
        del proposal
        machine.learning_enabled = True
        machine.sample = False
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed, learn=True, sample=False
        )
        machine.learning_enabled = False
        machine.sample = False
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    def evaluate(proposal):
        del proposal
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed + 1, learn=False, sample=False
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    # One observation pass, before any verifier outcome is read.
    templates = observed_templates(encoders, config, seed=seed)
    proposals = propose_from_bank(bank, templates)
    observation_passes = 1
    equivalence = None
    probe_payload: dict[str, Any] | None = None
    ranking: tuple[tuple[int, float], ...] = ()
    if proposer in ("dedup", "feedback"):
        # Signatures for dedup are computed on the stream the winner will be
        # scored on, which is what makes the collapse lossless rather than
        # approximate. Reading the stimulus costs environment time; it reads
        # no reward.
        evaluation_stream = observe_stream(encoders, config, seed=seed + 1)
        observation_passes += 1
        equivalence = partition_by_behaviour(
            proposals, machine, bank, evaluation_stream, install=install_proposal
        )
    probe_cost = 0
    if proposer == "feedback":
        # One episode, with whatever program happens to be installed first,
        # inverted into the target it was being scored against.
        install_proposal(machine, bank, proposals[equivalence.representatives[0]])
        probe_lifetime = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed, learn=False, sample=False
        )
        probe_cost = 1
        probe = probe_target(
            probe_lifetime,
            probe_label=proposals[equivalence.representatives[0]].label(),
        )
        probe_stream = observe_stream(encoders, config, seed=seed)
        observation_passes += 1
        rankable = tuple(
            index
            for index in equivalence.representatives
            if index not in equivalence.trained
        )
        signatures = signatures_for(
            [proposals[index] for index in rankable],
            machine,
            bank,
            probe_stream,
            install=install_proposal,
        )
        ranking = tuple(
            (rankable[local], score)
            for local, score in rank_by_agreement(signatures, probe)
        )
        probe_payload = probe.payload()
        probe_payload["ranked_candidates"] = len(ranking)
        probe_payload["best_probe_agreement"] = ranking[0][1] if ranking else None
        # A candidate whose probe accuracy is statistically incompatible with
        # a true rate at the gate cannot be worth an episode. This is the same
        # test the leases use to refuse a near-miss, applied before spending
        # rather than after, and it can discard a genuine winner only with
        # probability alpha.
        survivors = tuple(
            (index, score)
            for index, score in ranking
            if not control_below_threshold_report(
                score, probe.trials, threshold=THRESHOLD
            )["ruled_out_at_threshold"]
        )
        probe_payload["ruled_out_before_spending"] = len(ranking) - len(survivors)
        probe_payload["survivors"] = len(survivors)
        # A proposal that trains before it is scored has no signature to rank
        # or rule out. It cannot be discarded on evidence that predates its
        # training, so it goes to the back of the queue and is tried only if
        # nothing ranked gates first.
        trained = tuple(equivalence.trained)
        probe_payload["untestable_until_trained"] = len(trained)
        ranking = survivors + tuple((index, None) for index in trained)
    search = search_temporal_programs(
        bank,
        machine,
        evaluate,
        threshold=THRESHOLD,
        minimum_bits=MINIMUM_BITS,
        acquire=acquire,
        encoders=encoders,
        templates=templates,
        equivalence=equivalence,
        # An empty ranking means the probe ruled every candidate out, which is
        # a decision to spend nothing -- not an absent ranking. It must not
        # fall back to walking the whole list.
        order=(
            tuple(index for index, _ in ranking)
            if proposer == "feedback"
            else None
        ),
    )
    winner = search["winner"]
    executed = int(search["executed"]) + probe_cost
    row: dict[str, Any] = {
        "rule_digest": rule.digest(),
        "state_count": rule.state_count,
        "positive_rate": positive_rate(rule, seed=seed),
        "library_size": bank.program_count,
        "library_grown_by": bank.program_count - founding_program_count,
        "proposals_offered": len(proposals),
        "library_prefix": _library_prefix_length(proposals),
        "proposer": proposer,
        "observation_passes": observation_passes,
        "distinct_behaviours": search["distinct_behaviours"],
        "collapsed_as_equivalent": int(search["collapsed"]),
        "probe": probe_payload,
        "programs_executed": executed,
        "verifier_steps_spent": executed * steps,
        "solved": winner is not None,
        "winner_kind": winner["kind"] if winner else None,
        "winner_label": winner["label"] if winner else None,
        "winner_accuracy": float(winner["accuracy"]) if winner else None,
        "reused_learned_slot": bool(
            winner
            and winner["slots"]
            and max(winner["slots"]) >= founding_program_count
        ),
        "admitted": False,
        "admission_reason": None,
        "storable": None,
        "reproduces": None,
    }
    if winner is None:
        return row

    # Re-derive the winner in isolation so the machine holds exactly the state
    # that produced it, then ask it to reproduce on episodes it did not gate
    # on. Both costs are counted.
    proposal = _find_proposal(proposals, str(winner["label"]))
    if proposal is None:
        row["admission_reason"] = "winning proposal could not be re-derived"
        row["storable"] = False
        return row
    install_proposal(machine, bank, proposal)
    if proposal.kind in {"invent", "and"} and proposal.template is None:
        acquire(proposal)
    confirmations = []
    confirmation_trials = []
    for offset in range(CONFIRMATION_EPISODES):
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=seed + 100 + offset,
            learn=False,
            sample=False,
        )
        confirmations.append(float(report.eligible_accuracy))
        confirmation_trials.append(int(report.unique_verifier_bits))
    row["programs_executed"] += 1 + CONFIRMATION_EPISODES
    row["verifier_steps_spent"] = row["programs_executed"] * steps
    row["confirmation_accuracies"] = confirmations
    row["confirmation_trials"] = confirmation_trials

    # Gating once at 0.8 is not the same as being a 0.8 policy: the lease
    # machinery exists because a near-miss clears a single episode often
    # enough to matter. Pool the episodes the winner did not gate on and ask
    # whether its true rate can be ruled out as sitting at the gate.
    pooled_trials = sum(confirmation_trials)
    pooled_hits = sum(
        round(rate * trials)
        for rate, trials in zip(confirmations, confirmation_trials)
    )
    pooled_rate = pooled_hits / pooled_trials if pooled_trials else 0.0
    reproduction = control_below_threshold_report(
        pooled_rate,
        pooled_trials,
        threshold=THRESHOLD,
        control_label=str(winner["label"]),
    )
    row["pooled_confirmation_accuracy"] = pooled_rate
    row["reproduction"] = reproduction
    row["reproduces"] = not bool(reproduction["ruled_out_at_threshold"])

    artifact, delay_slot, reason = _winner_artifact(
        proposal, machine, bank, frontend_digest=encoders.digest()
    )
    row["storable"] = artifact is not None or proposal.kind == "retrieve"
    row["admission_reason"] = reason
    if artifact is None:
        return row

    if compression_admission:
        # DreamCoder's rule, in the only currency this family has: a file may
        # enter the library when it does something no existing file already
        # does. A behavioural duplicate lengthens the description of the
        # corpus without shortening any solution in it, and the first curve
        # measured what carrying such files costs.
        duplicate = _behavioural_duplicate(
            machine, bank, proposal, encoders, config, seed=seed + 1
        )
        row["compression_admission"] = True
        if duplicate is not None:
            row["admission_reason"] = (
                f"{reason}: rejected, presses identically to slot {duplicate}"
            )
            row["rejected_as_duplicate"] = True
            return row
        row["rejected_as_duplicate"] = False

    events = observe_events(encoders, config, seed=seed)
    context = events.mean(dim=0).detach().cpu().reshape(-1)
    try:
        if delay_slot is None:
            receipt = admit_temporal_program(
                bank_path,
                artifact,
                context,
                confirmations,
                threshold=THRESHOLD,
                min_observations=CONFIRMATION_EPISODES,
                min_stable_observations=1,
            )
        else:
            receipt = admit_and_child(
                bank_path,
                delay_slot,
                machine.prototype.detach().cpu().reshape(-1),
                context,
                confirmations,
                frontend_digest=encoders.digest(),
                threshold=THRESHOLD,
                min_observations=CONFIRMATION_EPISODES,
                min_stable_observations=1,
            )
    except (ValueError, RuntimeError) as error:
        row["admission_reason"] = f"{reason}: rejected by the bank ({error})"
        row["storable"] = False
        return row
    row["admitted"] = bool(receipt.accepted)
    row["admission_slot"] = None if receipt.slot is None else int(receipt.slot)
    if not receipt.accepted:
        row["admission_reason"] = f"{reason}: {receipt.reason}"
    return row


def _behavioural_duplicate(
    machine,
    bank: ExternalTemporalProgramBank,
    proposal: ProgramProposal,
    encoders,
    config: RenderedBrainWorkshopConfig,
    *,
    seed: int,
) -> int | None:
    """The slot a candidate would merely restate, if there is one.

    Compares presses on one observation pass. Reads no reward, so a candidate
    can be rejected as redundant without spending anything to find out.
    """

    stream = observe_stream(encoders, config, seed=seed)
    install_proposal(machine, bank, proposal)
    candidate = behaviour_signature(machine, stream)
    for slot in range(bank.program_count):
        try:
            install_proposal(
                machine, bank, ProgramProposal("retrieve", (slot,), bank.artifact(slot))
            )
        except (ValueError, RuntimeError):
            # A slot this machine cannot host cannot be the file a candidate
            # duplicates, so it simply does not compete.
            continue
        if behaviour_signature(machine, stream) == candidate:
            return slot
    install_proposal(machine, bank, proposal)
    return None


def _copy_bank(source: Path, destination: Path) -> None:
    """Copy a bank and the independent checksum it refuses to load without."""

    shutil.copyfile(source, destination)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    shutil.copyfile(sidecar, destination.with_suffix(destination.suffix + ".sha256"))


def run_arm(
    payload: dict[str, object],
    encoders,
    rules,
    founding_bank: Path,
    scratch: Path,
    *,
    grow: bool,
    seed: int,
    steps: int,
    proposer: str = "enumerate",
    compression_admission: bool = False,
) -> dict[str, Any]:
    """One curriculum pass. `grow=False` restores the library every rule."""

    _copy_bank(founding_bank, scratch)
    founding_count = ExternalTemporalProgramBank.load_bank(scratch).program_count
    rows: list[dict[str, Any]] = []
    for rule in rules:
        if not grow:
            _copy_bank(founding_bank, scratch)
        rows.append(
            _learn_one_rule(
                payload,
                scratch,
                encoders,
                rule,
                seed=seed,
                steps=steps,
                founding_program_count=founding_count,
                proposer=proposer,
                compression_admission=compression_admission,
            )
        )
    if not grow:
        # The last rule admitted into the scratch copy like any other. Restore
        # once more so the recorded final count means what it says: a control
        # agent ends where it started.
        _copy_bank(founding_bank, scratch)
    final = ExternalTemporalProgramBank.load_bank(scratch)
    return {
        "grew_the_library": grow,
        "proposer": proposer,
        "compression_admission": compression_admission,
        "founding_program_count": founding_count,
        "final_program_count": final.program_count,
        "solved": sum(1 for row in rows if row["solved"]),
        "reproduced": sum(1 for row in rows if row["reproduces"]),
        "of": len(rows),
        "total_programs_executed": sum(int(row["programs_executed"]) for row in rows),
        "total_verifier_steps": sum(int(row["verifier_steps_spent"]) for row in rows),
        "rules": rows,
    }


def _curve(growing: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-rule cost of the growing arm against its matched control."""

    points = []
    for index, (grown, fixed) in enumerate(zip(growing["rules"], control["rules"])):
        if grown["rule_digest"] != fixed["rule_digest"]:
            raise RuntimeError("arms disagree about the curriculum order")
        points.append(
            {
                "position": index,
                "rule_digest": grown["rule_digest"],
                "state_count": grown["state_count"],
                "library_size": grown["library_size"],
                "growing_cost": grown["programs_executed"],
                "control_cost": fixed["programs_executed"],
                "cost_ratio": (
                    grown["programs_executed"] / fixed["programs_executed"]
                    if fixed["programs_executed"]
                    else None
                ),
                "growing_solved": grown["solved"],
                "control_solved": fixed["solved"],
                "growing_reproduces": grown["reproduces"],
                "control_reproduces": fixed["reproduces"],
                "reused_learned_slot": grown["reused_learned_slot"],
            }
        )
    return points


def run_accumulation_curve(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    scratch_directory: Path,
    frontend_path: Path | None = None,
    state_counts: tuple[int, ...] = STATE_COUNTS,
    rules_per_state_count: int = RULES_PER_STATE_COUNT,
    symbol_count: int = SYMBOL_COUNT,
    steps: int = STEPS,
    seed: int = DEVELOPMENT_SEED,
    proposer: str = "enumerate",
    compression_admission: bool = False,
) -> dict[str, Any]:
    """Both arms, the curve between them, and a receipt that nothing moved."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    rules = curriculum_rules(
        state_counts=state_counts,
        rules_per_state_count=rules_per_state_count,
        symbol_count=symbol_count,
    )
    scratch_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    growing = run_arm(
        payload,
        encoders,
        rules,
        bank_path,
        scratch_directory / "growing.bank",
        grow=True,
        seed=seed,
        steps=steps,
        proposer=proposer,
        compression_admission=compression_admission,
    )
    control = run_arm(
        payload,
        encoders,
        rules,
        bank_path,
        scratch_directory / "control.bank",
        grow=False,
        seed=seed,
        steps=steps,
        proposer=proposer,
        compression_admission=compression_admission,
    )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("accumulation curve mutated AgentBrain.bank")
    curve = _curve(growing, control)
    unstorable = [
        row
        for row in growing["rules"]
        if row["solved"] and row["storable"] is False
    ]
    report = {
        "schema": ACCUMULATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": (
            "development seed, already consumed; the curriculum grows a scratch "
            "copy and AgentBrain.bank is never written"
        ),
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "steps": steps,
        "symbol_count": symbol_count,
        "threshold": THRESHOLD,
        "curriculum_length": len(rules),
        "proposer": proposer,
        "compression_admission": compression_admission,
        "growing": growing,
        "control": control,
        "curve": curve,
        "accumulated_slots": (
            growing["final_program_count"] - growing["founding_program_count"]
        ),
        "solved_winners_the_bank_cannot_hold": len(unstorable),
        "cost_ratio_overall": (
            growing["total_programs_executed"] / control["total_programs_executed"]
            if control["total_programs_executed"]
            else None
        ),
        "any_rule_reused_a_learned_slot": any(
            row["reused_learned_slot"] for row in growing["rules"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "accumulation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    checksums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_directory.glob("*.json"))
    ]
    (output_directory / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-artifact",
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
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_accumulation_curve_2026-08-15"
        ),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=repository / "artifacts/scratch/accumulation_curve",
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--proposer",
        choices=("enumerate", "dedup", "feedback"),
        default="enumerate",
        help="how candidates are ordered and filtered",
    )
    parser.add_argument(
        "--compression-admission",
        action="store_true",
        help="admit a file only if no existing file already presses that way",
    )
    arguments = parser.parse_args()
    report = run_accumulation_curve(
        arguments.controller_artifact,
        arguments.bank,
        arguments.output_dir,
        scratch_directory=arguments.scratch_dir,
        frontend_path=arguments.frontend,
        steps=arguments.steps,
        proposer=arguments.proposer,
        compression_admission=arguments.compression_admission,
    )
    print(
        json.dumps(
            {
                "bank_unchanged": report["bank_unchanged"],
                "growing": {
                    "solved": f"{report['growing']['solved']}/{report['growing']['of']}",
                    "executed": report["growing"]["total_programs_executed"],
                },
                "control": {
                    "solved": f"{report['control']['solved']}/{report['control']['of']}",
                    "executed": report["control"]["total_programs_executed"],
                },
                "cost_ratio_overall": report["cost_ratio_overall"],
                "accumulated_slots": report["accumulated_slots"],
                "any_rule_reused_a_learned_slot": report[
                    "any_rule_reused_a_learned_slot"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
