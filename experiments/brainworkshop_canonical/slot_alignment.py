"""Which thing is which, when nothing stays still.

The object agent worked out its own marker with a rule that reads well and does
not survive contact: *the goal is the thing that never moved*. It is true only
because exactly one thing in the scene was static. Add a second moving object
and the rule does not degrade -- it inverts, because the intersection over
observations is now empty or, worse, accidentally a singleton that is wrong.

DeepMind named this the correspondence problem and attacked it with AlignNet
(Creswell et al., 2020): segmentation tells you *what is in* frame t and *what
is in* frame t+1, and nothing at all about which is which. Their Memory variant
aligns against a persistent slot memory rather than against the previous frame,
so an object that disappears and comes back is the same object. The network is
trained; the inductive bias is not, and the bias is the part that transfers to a
repository with no gradients in the agent path.

So there are two mechanisms here, and they answer different halves.

**Alignment gives persistence.** A fixed set of tracks is matched to this
frame's parts. The matching is a *function* from tracks to parts rather than a
permutation, because two markers standing on one place is one part -- objects
here merge and separate rather than occlude, and a permutation cannot express
that. Every part must receive at least one track, or something appeared from
nowhere.

What the matching is scored by took three tries, and the failures are the
useful part.

- **Displacement**, which is what AlignNet uses, assumes things move a little
  between frames. This world teleports: an action sends the agent to an
  arbitrary place, so the nearest centroid is frequently the other object.
- **Minimum change** replaces position with symbol -- one action moves one
  thing, so prefer the correspondence under which fewest tracks moved. Correct
  with two objects. With three it is simply false, because two things move at
  every step, and measured, it followed the agent 57% of the time and did not
  improve with eight times the experience.
- **Search** keeps several correspondences alive and scores each by how
  cheaply the per-track tables it implies encode what actually happened. The
  agent's own marker is the only one whose behaviour is a consistent function
  of place and action, so the right correspondence is the one that makes some
  track predictable. This is the gradient-free relative of the
  expectation-maximisation trackers, and it is the only version that works
  when more than one thing moves.

**Dynamics give identity.** Persistence says a track is the same thing over
time; it does not say which thing. That is settled by asking what the actions
do, which is evidence the agent generates rather than receives:

- a track whose successor depends on the action is the one the agent is
  driving -- itself;
- a track that never changes symbol is a fixed target;
- a track that moves but whose successor does not depend on the action is
  something else moving in the world, and the agent does not control it.

The third case is the one the persistence rule cannot represent at all, and it
is why controllability rather than motion is the discriminator. A distractor
walking a fixed cycle is perfectly predictable from its own state; what it is
not is *responsive*. See `TrackEvidence.controllability` for why that has to be
measured as a matched contrast rather than as accuracy or as description
length, both of which were tried and both of which fail here for opposite
reasons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

SLOT_ALIGNMENT_SCHEMA = "neural-computer.slot-alignment.v1"

# Enumerating every track-to-part map costs parts ** tracks. Scenes here hold a
# handful of markers; the guard is so that a caller who wires this to a richer
# frontend gets an error rather than a hang.
MAXIMUM_TRACKS = 6


def _square_distance(left, right) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def assign(costs: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Cheapest map from each track to a part, covering every part.

    Exhaustive rather than Hungarian, because it is not a permutation: several
    tracks may share a part when the things they follow coincide. Hungarian
    solves the injective problem, which is the wrong problem for objects that
    merge.
    """

    tracks = len(costs)
    if tracks < 1:
        raise ValueError("alignment needs at least one track")
    if tracks > MAXIMUM_TRACKS:
        raise ValueError("too many tracks to align exhaustively")
    parts = len(costs[0])
    if parts < 1:
        raise ValueError("alignment needs at least one part")
    if any(len(row) != parts for row in costs):
        raise ValueError("the cost matrix is ragged")
    if parts > tracks:
        raise ValueError("more parts than tracks: something appeared from nowhere")
    best: tuple[int, ...] | None = None
    best_cost = float("inf")
    for candidate in product(range(parts), repeat=tracks):
        if len(set(candidate)) != parts:
            # Some part received no track, so a marker would have to have
            # arrived without ever having been anywhere.
            continue
        cost = sum(costs[track][part] for track, part in enumerate(candidate))
        if cost < best_cost:
            best_cost = cost
            best = candidate
    if best is None:  # pragma: no cover - excluded by the parts <= tracks check
        raise ValueError("no assignment covers every part")
    return best


def displacement_costs(track_centroids: Sequence, part_centroids: Sequence):
    """How far each track would have to have moved to be each part."""

    return [
        [_square_distance(track, part) for part in part_centroids]
        for track in track_centroids
    ]


# Matching by displacement assumes things move a little between frames.
# Measured here, that assumption is simply false: an action in this world sends
# the agent to an arbitrary place, so a marker can cross the grid in one step
# and the nearest centroid is often the *other* object. What the world is
# smooth in is not position but symbol -- one action moves one thing -- so the
# dominant term is the number of tracks whose place changes.
SYMBOL_CHANGE_WEIGHT = 1.0e3
# Under that sits AlignNet's actual contribution: align against the slot
# *memory* rather than against the previous frame. It earns its place at one
# specific moment. When two markers coincide and then separate, both tracks
# last saw the same symbol, so minimum-change cannot say which of them moved
# off, and the tie was being broken by a coin. A track that has spent the
# episode at one place is the thing that is still there.
MEMORY_WEIGHT = 1.0
# Displacement survives only as the last tie-break, scaled below everything
# else rather than deleted: it is weak evidence here, not absent.
DISPLACEMENT_SCALE = 1.0e-4
# Above all of it sits the track's own model of itself, once it has one. This
# outranks minimum-change because minimum-change has a failure it cannot see:
# if one marker moves off a place and another moves onto it, the cheapest
# explanation is that neither moved and the two swapped identities. A track
# that already knows where this action takes it is not fooled. Applied only
# where the track's table is confident, so it contributes nothing early and
# decides late.
PREDICTION_WEIGHT = 1.0e4
CONFIDENT_REPEATS = 2
# How much an unpredictable outcome is worth to a probe, against a unit of
# actual disagreement between surviving correspondences.
UNTRIED_INFORMATION = 0.5


def persistence_costs(
    track_centroids: Sequence,
    track_symbols: Sequence,
    part_centroids: Sequence,
    part_symbols: Sequence,
    track_memories: Sequence | None = None,
    predictions: Sequence | None = None,
):
    """Prefer the correspondence under which the fewest things moved.

    `track_memories` are per-track counts of the symbols that track has stood
    on. `predictions` are per-track guesses at where this action should have
    taken it, or `None` where the track has no confident opinion. Where both
    are absent this is pure minimum-change, which is the right behaviour for
    the first frame, when there is nothing else to go on.
    """

    costs = []
    for index, (centroid, symbol) in enumerate(zip(track_centroids, track_symbols)):
        memory = None if track_memories is None else track_memories[index]
        expected = None if predictions is None else predictions[index]
        seen = sum(memory.values()) if memory else 0
        row = []
        for part_centroid, part_symbol in zip(part_centroids, part_symbols):
            familiarity = (
                (memory.get(int(part_symbol), 0) / seen) if memory and seen else 0.0
            )
            surprise = (
                0.0
                if expected is None or int(expected) == int(part_symbol)
                else PREDICTION_WEIGHT
            )
            row.append(
                surprise
                + SYMBOL_CHANGE_WEIGHT * float(int(symbol) != int(part_symbol))
                + MEMORY_WEIGHT * (1.0 - familiarity)
                + DISPLACEMENT_SCALE * _square_distance(centroid, part_centroid)
            )
        costs.append(row)
    return costs


@dataclass
class Tracker:
    """A persistent slot memory, updated by alignment rather than by index.

    Started from a frame in which every object is separately visible, which is
    the only moment the count is unambiguous. After that the count is held
    fixed and merges are absorbed by the assignment, so a track survives its
    object being hidden behind another.
    """

    centroids: list[tuple[float, float]]
    symbols: list[int]
    memories: list[dict[int, int]] = field(default_factory=list)
    # Per track: (symbol, action) -> counts of what followed. The track's own
    # model of itself, used to keep hold of it when several things move at once.
    tables: list[dict[tuple[int, int], dict[int, int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        while len(self.memories) < len(self.symbols):
            self.memories.append({self.symbols[len(self.memories)]: 1})
        while len(self.tables) < len(self.symbols):
            self.tables.append({})

    def _prediction(self, track: int, action: int | None) -> int | None:
        """Where this track should be, if it is confident and consistent."""

        if action is None:
            return None
        counts = self.tables[track].get((int(self.symbols[track]), int(action)))
        if not counts:
            return None
        best, seen = max(counts.items(), key=lambda item: item[1])
        if seen < CONFIDENT_REPEATS or seen != sum(counts.values()):
            return None
        return int(best)

    @classmethod
    def started(cls, parts: Sequence) -> Tracker:
        if not parts:
            raise ValueError("a tracker needs at least one part to start from")
        if len(parts) > MAXIMUM_TRACKS:
            raise ValueError("too many parts to track")
        return cls(
            centroids=[tuple(centroid) for centroid, _ in parts],
            symbols=[int(symbol) for _, symbol in parts],
        )

    @property
    def count(self) -> int:
        return len(self.centroids)

    def update(self, parts: Sequence, *, action: int | None = None) -> tuple[int, ...]:
        """Take one frame's parts; return the part each track was matched to.

        More parts than tracks means something separated that had been hidden,
        or arrived. Rather than fail, the surplus becomes new tracks -- the
        memory grows and never shrinks, which is AlignNet's persistence bias:
        once a thing has been seen it continues to exist even while it cannot
        be seen.
        """

        if not parts:
            raise ValueError("a scene must contain at least one part")
        centroids = [tuple(centroid) for centroid, _ in parts]
        symbols = [int(symbol) for _, symbol in parts]
        spawned: set[int] = set()
        while len(centroids) > self.count:
            # The part least explained by anything already remembered.
            unclaimed = max(
                (part for part in range(len(centroids)) if part not in spawned),
                key=lambda part: min(
                    _square_distance(known, centroids[part]) for known in self.centroids
                ),
            )
            spawned.add(unclaimed)
            self.centroids.append(centroids[unclaimed])
            self.symbols.append(symbols[unclaimed])
            self.memories.append({symbols[unclaimed]: 1})
            self.tables.append({})
        predictions = [
            self._prediction(track, action) for track in range(self.count)
        ]
        mapping = assign(
            persistence_costs(
                self.centroids,
                self.symbols,
                centroids,
                symbols,
                self.memories,
                predictions,
            )
        )
        previous = list(self.symbols)
        for track, part in enumerate(mapping):
            self.centroids[track] = centroids[part]
            self.symbols[track] = symbols[part]
            memory = self.memories[track]
            memory[symbols[part]] = memory.get(symbols[part], 0) + 1
            if action is not None:
                cell = self.tables[track].setdefault(
                    (int(previous[track]), int(action)), {}
                )
                cell[symbols[part]] = cell.get(symbols[part], 0) + 1
        return mapping

    def reading(self) -> tuple[int, ...]:
        """The symbol each track currently stands on."""

        return tuple(self.symbols)


# --- identity, from what the actions do ------------------------------------


@dataclass(frozen=True)
class TrackEvidence:
    """What one track did, and how much of it the agent was responsible for."""

    moved: float
    with_action: float
    without_action: float
    with_bits: float
    without_bits: float
    steps: int

    # How often leaving the same place by *different* actions led somewhere
    # different, and how often leaving it by the *same* action did. The second
    # is the noise floor for the first.
    action_effect: float
    repeat_disagreement: float
    contrasts: int

    @property
    def controllability(self) -> float:
        """How much the action changes where this track ends up.

        Two earlier versions of this were wrong in instructive ways.

        Accuracy, fitted and scored on one short stream, says a marker moving
        at *random* is perfectly predictable, because a table keyed by
        `(symbol, action)` sees almost every key once.

        Description length is not fooled by that, but it cannot be used here
        either: with twenty steps and thirty-two possible keys, conditioning on
        the action can never repay the table it costs, so every track -- the
        agent included -- scores negative and the comparison says nothing. That
        is MDL being right about the evidence and useless as a discriminator.

        What works in this sample size is a *matched* contrast. Among visits to
        the same place, compare what happened under different actions against
        what happened under the same action twice. The second is the noise
        floor, so their difference isolates the action's effect and needs only
        a handful of repeated visits rather than a populated table.

        A marker moving at random scores near zero because both rates are
        high; one on a fixed circuit scores near zero because both are low;
        the agent's own marker is the one where they come apart.
        """

        return self.action_effect - self.repeat_disagreement

    def payload(self) -> dict[str, Any]:
        return {
            "moved": self.moved,
            "with_action": self.with_action,
            "without_action": self.without_action,
            "with_bits": self.with_bits,
            "without_bits": self.without_bits,
            "action_effect": self.action_effect,
            "repeat_disagreement": self.repeat_disagreement,
            "contrasts": self.contrasts,
            "steps": self.steps,
            "controllability": self.controllability,
        }


def _matched_contrast(steps: Sequence[tuple[int, int, int]]) -> tuple[float, float, int]:
    """Within each place, how often the successor differed.

    Split by whether the two visits used different actions or the same one.
    Comparing only visits to the *same* place is what makes this work on short
    streams: it never has to estimate a distribution, only to notice that two
    matched attempts came out differently.
    """

    grouped: dict[int, list[tuple[int, int]]] = {}
    for symbol, action, following in steps:
        grouped.setdefault(int(symbol), []).append((int(action), int(following)))
    differing = same = 0
    differing_disagree = same_disagree = 0
    for visits in grouped.values():
        for index, (action, following) in enumerate(visits):
            for other_action, other_following in visits[index + 1 :]:
                disagree = following != other_following
                if action == other_action:
                    same += 1
                    same_disagree += int(disagree)
                else:
                    differing += 1
                    differing_disagree += int(disagree)
    return (
        differing_disagree / differing if differing else 0.0,
        same_disagree / same if same else 0.0,
        differing + same,
    )


def _grouped(rows: Sequence[tuple[tuple, int]]) -> dict[tuple, dict[int, int]]:
    table: dict[tuple, dict[int, int]] = {}
    for key, following in rows:
        table.setdefault(key, {})
        table[key][following] = table[key].get(following, 0) + 1
    return table


def _table_accuracy(rows: Sequence[tuple[tuple, int]]) -> float:
    """Accuracy of the best deterministic table over the given keys."""

    if not rows:
        return 0.0
    table = _grouped(rows)
    return sum(max(counts.values()) for counts in table.values()) / len(rows)


def _table_bits(rows: Sequence[tuple[tuple, int]], alphabet: int) -> float:
    """Bits to write the table down, plus bits for what it still gets wrong."""

    if not rows:
        return 0.0
    width = math.log2(max(2, int(alphabet)))
    table = _grouped(rows)
    description = len(table) * width
    mistakes = sum(
        sum(counts.values()) - max(counts.values()) for counts in table.values()
    )
    return description + mistakes * width


@dataclass
class TrackHistory:
    """One track's symbols, the actions taken beside them, and what followed."""

    steps: list[tuple[int, int, int]] = field(default_factory=list)

    def observe(self, symbol: int, action: int, following: int) -> None:
        self.steps.append((int(symbol), int(action), int(following)))

    def evidence(self, *, alphabet: int | None = None) -> TrackEvidence:
        if not self.steps:
            return TrackEvidence(
                moved=0.0,
                with_action=0.0,
                without_action=0.0,
                with_bits=0.0,
                without_bits=0.0,
                action_effect=0.0,
                repeat_disagreement=0.0,
                contrasts=0,
                steps=0,
            )
        moved = sum(
            1 for symbol, _, following in self.steps if symbol != following
        ) / len(self.steps)
        keyed = [
            ((symbol, action), following) for symbol, action, following in self.steps
        ]
        plain = [((symbol,), following) for symbol, _, following in self.steps]
        width = (
            alphabet
            if alphabet is not None
            else 1 + max(
                max(symbol, following) for symbol, _, following in self.steps
            )
        )
        effect, repeats, contrasts = _matched_contrast(self.steps)
        return TrackEvidence(
            moved=moved,
            with_action=_table_accuracy(keyed),
            without_action=_table_accuracy(plain),
            with_bits=_table_bits(keyed, width),
            without_bits=_table_bits(plain, width),
            action_effect=effect,
            repeat_disagreement=repeats,
            contrasts=contrasts,
            steps=len(self.steps),
        )


def _surjections(tracks: int, parts: int):
    """Every map from tracks onto parts that leaves no part unexplained."""

    for candidate in product(range(parts), repeat=tracks):
        if len(set(candidate)) == parts:
            yield candidate


def _code_length(counts: dict[int, int], symbol: int, alphabet: int) -> float:
    """Prequential cost of this successor, given what the table has seen.

    Laplace-smoothed and charged *before* the observation is added, so a track
    whose behaviour is consistent gets cheaper over the episode and one whose
    behaviour is arbitrary never does. Nothing is fitted in advance: the total
    is the cost of transmitting the stream to someone who is learning the same
    table as they go, which is what makes this comparable across hypotheses
    that disagree about which object is which.
    """

    total = sum(counts.values())
    seen = counts.get(int(symbol), 0)
    return -math.log2((seen + 0.5) / (total + 0.5 * max(2, int(alphabet))))


@dataclass
class _Hypothesis:
    symbols: tuple[int, ...]
    tables: tuple[dict[tuple[int, int], dict[int, int]], ...]
    bits: float
    trace: tuple[tuple[int, ...], ...]


def beam_hypotheses(
    readings: Sequence[Sequence[tuple[Any, int]]],
    actions: Sequence[int],
    *,
    alphabet: int,
    beam: int = 8,
) -> list[_Hypothesis]:
    """The surviving correspondences, cheapest first, with their code lengths.

    `beam_track` throws all but the best away. Keeping them is what makes it
    possible to *act* on the ambiguity rather than wait it out: two
    correspondences that disagree about which marker the agent is will often
    disagree about what a particular action does next, and that is a question
    the agent can go and settle.
    """

    if not readings:
        return []
    width = max(len(frame) for frame in readings)
    if width > MAXIMUM_TRACKS:
        raise ValueError("too many parts to track")
    first = list(readings[0])
    hypotheses: list[_Hypothesis] = []
    for mapping in _surjections(width, len(first)):
        symbols = tuple(int(first[part][1]) for part in mapping)
        hypotheses.append(
            _Hypothesis(
                symbols=symbols,
                tables=tuple({} for _ in range(width)),
                bits=0.0,
                trace=(symbols,),
            )
        )

    for step, frame in enumerate(readings[1:]):
        action = int(actions[step]) if step < len(actions) else 0
        parts = [int(symbol) for _, symbol in frame]
        expanded: list[_Hypothesis] = []
        for hypothesis in hypotheses:
            for mapping in _surjections(width, len(parts)):
                cost = 0.0
                tables = []
                symbols = []
                for track, part in enumerate(mapping):
                    key = (int(hypothesis.symbols[track]), action)
                    counts = hypothesis.tables[track].get(key, {})
                    following = parts[part]
                    cost += _code_length(counts, following, alphabet)
                    updated = dict(hypothesis.tables[track])
                    cell = dict(counts)
                    cell[following] = cell.get(following, 0) + 1
                    updated[key] = cell
                    tables.append(updated)
                    symbols.append(following)
                expanded.append(
                    _Hypothesis(
                        symbols=tuple(symbols),
                        tables=tuple(tables),
                        bits=hypothesis.bits + cost,
                        trace=hypothesis.trace + (tuple(symbols),),
                    )
                )
        expanded.sort(key=lambda item: item.bits)
        hypotheses = expanded[: int(beam)]
    return hypotheses


def hypothesis_weights(hypotheses: Sequence[_Hypothesis]) -> list[float]:
    """Posterior over correspondences, from their code lengths."""

    if not hypotheses:
        return []
    best = min(item.bits for item in hypotheses)
    unnormalised = [2.0 ** (-(item.bits - best)) for item in hypotheses]
    total = sum(unnormalised) or 1.0
    return [value / total for value in unnormalised]


def disambiguating_action(
    hypotheses: Sequence[_Hypothesis], *, action_count: int
) -> int | None:
    """The action whose outcome the surviving correspondences most disagree on.

    Waiting for the ambiguity to resolve itself is measurably the wrong move --
    a longer orientation costs more than the better answer buys. The
    alternative is to *design the next probe*: score each action by how much
    the hypotheses disagree about the reading it will produce, and take the one
    they disagree on most.

    An action that every surviving correspondence expects to look the same
    afterwards teaches nothing, however novel the place it leads to. This is
    the only place in the repository where an action is chosen for what it will
    *reveal* rather than for what it will earn.

    Returns nothing when no action separates anything -- either because the
    hypotheses already agree, or because none of them has an opinion yet.
    """

    if not hypotheses or action_count < 1:
        return None
    weights = hypothesis_weights(hypotheses)
    best_action: int | None = None
    best_score = 0.0
    for action in range(int(action_count)):
        predicted: dict[tuple[int, ...], float] = {}
        informed = 0.0
        unknown = 0.0
        for hypothesis, weight in zip(hypotheses, weights):
            reading = []
            known = True
            for track, symbol in enumerate(hypothesis.symbols):
                counts = hypothesis.tables[track].get((int(symbol), action))
                if not counts:
                    known = False
                    break
                reading.append(max(counts.items(), key=lambda item: item[1])[0])
            if not known:
                # An outcome nobody can predict is not a disagreement, but it
                # is the only thing that *creates* one later. Without this term
                # the probe is silent for exactly the steps it is needed --
                # measured, it returned nothing for the first seven of twelve,
                # by which point orientation had already happened.
                unknown += weight
                continue
            informed += weight
            key = tuple(sorted(reading))
            predicted[key] = predicted.get(key, 0.0) + weight
        # Expected disagreement: one minus the chance two draws from the
        # predicted-reading distribution come out the same.
        score = UNTRIED_INFORMATION * unknown
        if informed > 0.0:
            share = [value / informed for value in predicted.values()]
            score += informed * (1.0 - sum(value * value for value in share))
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def contrasting_action(
    hypotheses: Sequence[_Hypothesis], *, action_count: int
) -> int | None:
    """The action that creates a matched contrast, rather than a novel one.

    This is the opposite of a curiosity probe, and the failure of three
    curiosity-shaped probes is what argues for it. Controllability is measured
    as a *matched* contrast -- the same place left by two different actions --
    so what identification needs is not somewhere new but somewhere already
    visited, left differently this time. Every probe that sought novelty
    reduced the number of matched pairs and made identification worse.

    Scores each action by how many previous visits to the current symbol used
    a *different* action, summed over the surviving correspondences: taking it
    now completes that many contrasts at once.
    """

    if not hypotheses or action_count < 1:
        return None
    weights = hypothesis_weights(hypotheses)
    best_action: int | None = None
    best_score = 0.0
    for action in range(int(action_count)):
        score = 0.0
        for hypothesis, weight in zip(hypotheses, weights):
            for track, symbol in enumerate(hypothesis.symbols):
                completed = sum(
                    sum(counts.values())
                    for (key_symbol, key_action), counts
                    in hypothesis.tables[track].items()
                    if key_symbol == int(symbol) and key_action != action
                )
                score += weight * completed
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def model_consistency(
    history: TrackHistory, successor
) -> tuple[float, int]:
    """How much of this track's behaviour matches a model of *my* dynamics.

    The question every earlier mechanism asked was "which of these tracks looks
    controllable, judging only by this episode". That throws away the fact that
    the world is the same world it was last episode. A self-model already says
    where each action takes me; a track that agrees with it is me, and a track
    that does not is something else.

    `successor(symbol, action)` returns the believed next symbol, or `None`
    where the model has no opinion. Only the cells it has an opinion about are
    scored, so a sparse model abstains rather than guessing.
    """

    agreed = tested = 0
    for symbol, action, following in history.steps:
        believed = successor(symbol, action)
        if believed is None:
            continue
        tested += 1
        agreed += int(int(believed) == int(following))
    return (agreed / tested if tested else 0.0), tested


def identify_by_model(
    histories: Sequence[TrackHistory],
    successor,
    *,
    minimum_tested: int = 2,
    minimum_agreement: float = 0.6,
) -> int | None:
    """Which track is me, according to what I already know about myself.

    Refuses when the model has too little to say, so the first episode -- when
    there is no model yet -- falls back to whatever derived the identity before.
    """

    best: int | None = None
    best_score = 0.0
    for index, history in enumerate(histories):
        agreement, tested = model_consistency(history, successor)
        if tested < int(minimum_tested) or agreement < float(minimum_agreement):
            continue
        score = agreement * tested
        if score > best_score:
            best_score = score
            best = index
    return best


def self_log_evidence(
    history: TrackHistory, counts, *, alphabet: int, smoothing: float = 0.5
) -> float:
    """How probable this track's transitions are under my own dynamics.

    Replaces an agreement threshold with a likelihood, which is what lets
    evidence of different amounts and different quality be compared. A track
    seen twice and agreeing twice is not the same claim as one seen twenty
    times and agreeing eighteen, and a ratio cannot tell them apart.

    `counts(symbol, action)` returns the accumulated successor counts the agent
    has for itself, or nothing where it has none. Cells with no evidence
    contribute the uniform code, so a sparse model neither helps nor punishes.
    """

    width = max(2, int(alphabet))
    total = 0.0
    for symbol, action, following in history.steps:
        seen = counts(symbol, action) or {}
        observed = sum(seen.values())
        hits = seen.get(int(following), 0.0)
        total += math.log2(
            (hits + smoothing) / (observed + smoothing * width)
        )
    return total


def self_posterior(
    histories: Sequence[TrackHistory], counts, *, alphabet: int
) -> list[float]:
    """A belief over which track is me, rather than a decision.

    The hard version has a failure that is easy to walk into and hard to see:
    name the wrong track once, learn its dynamics, and that model then names it
    again. Measured, the hard loop reaches its fixed point after a single pass
    and never moves. A posterior lets an ambiguous episode contribute almost
    nothing to the self-model instead of contributing a confident mistake.
    """

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


def beam_track(
    readings: Sequence[Sequence[tuple[Any, int]]],
    actions: Sequence[int],
    *,
    alphabet: int,
    beam: int = 8,
) -> tuple[tuple[int, ...], ...]:
    """Correspondence by search, when committing frame by frame cannot work.

    Greedy alignment assumes one thing moves at a time. With a distractor two
    things move at every step, the assumption is simply false, and -- measured
    -- the greedy tracker follows the agent barely more than half the time and
    does not improve with eight times the experience. That is not sparsity; it
    is the prior being wrong.

    What is still true is that the agent's own marker is the only one whose
    behaviour is a *consistent function* of place and action. So instead of
    committing to a correspondence and then learning tables from it, this keeps
    several correspondences alive and scores each by how cheaply the tables it
    implies encode what actually happened. The right assignment is the one
    under which some track turns out to be predictable.

    This is the small, gradient-free relative of the expectation-maximisation
    trackers -- alternating between "which is which" and "what does each do"
    rather than solving either first.

    It has a measured limit: on a twelve-step history with two markers both
    moving it swaps tracks partway and returns a mixture, and it recovers the
    correspondence at twenty-four. `disambiguating_action` is the attempt to
    buy those steps back by choosing better ones rather than more.
    """

    hypotheses = beam_hypotheses(readings, actions, alphabet=alphabet, beam=beam)
    return hypotheses[0].trace if hypotheses else ()


@dataclass(frozen=True)
class Roles:
    """Which track the agent is, and which ones it has been asked to reach."""

    own: int | None
    targets: tuple[int, ...]
    loose: tuple[int, ...]
    evidence: tuple[TrackEvidence, ...]

    @property
    def target(self) -> int | None:
        """The single target, when the scene names exactly one."""

        return self.targets[0] if len(self.targets) == 1 else None

    def payload(self) -> dict[str, Any]:
        return {
            "own": self.own,
            "targets": list(self.targets),
            "loose": list(self.loose),
            "evidence": [item.payload() for item in self.evidence],
        }


def identify_roles(
    histories: Sequence[TrackHistory],
    *,
    control_margin: float = 0.1,
    motion_margin: float = 0.05,
    alphabet: int | None = None,
) -> Roles:
    """Sort the tracks into me, what I was asked for, and what is merely there.

    Two thresholds, both about refusing rather than guessing. A track is the
    agent's own only if knowing the action buys a real improvement, and a track
    is a target only if it genuinely never moved. Everything else is `loose`:
    moving, and not moving because of me. That third bin is the one the
    persistence rule has no room for, and having somewhere to put a distractor
    is most of why this works where that did not.
    """

    evidence = tuple(history.evidence(alphabet=alphabet) for history in histories)
    if not evidence:
        return Roles(own=None, targets=(), loose=(), evidence=())

    own: int | None = None
    movers = [
        index for index, item in enumerate(evidence) if item.moved > motion_margin
    ]
    if len(movers) == 1:
        # Only one thing in the scene is going anywhere, so no amount of
        # further evidence can change the answer and waiting for it is pure
        # cost. Measured: insisting on the controllability margin here spent
        # eleven of twenty steps deciding, which is most of the episode.
        own = movers[0]
    else:
        responsive = [
            index
            for index, item in enumerate(evidence)
            if item.controllability >= control_margin
        ]
        if responsive:
            own = max(responsive, key=lambda index: evidence[index].controllability)

    targets = tuple(
        index
        for index, item in enumerate(evidence)
        if item.moved <= motion_margin and index != own
    )
    loose = tuple(
        index
        for index in range(len(evidence))
        if index != own and index not in targets
    )
    return Roles(own=own, targets=targets, loose=loose, evidence=evidence)


__all__ = [
    "MAXIMUM_TRACKS",
    "SLOT_ALIGNMENT_SCHEMA",
    "SYMBOL_CHANGE_WEIGHT",
    "UNTRIED_INFORMATION",
    "Roles",
    "TrackEvidence",
    "TrackHistory",
    "Tracker",
    "assign",
    "beam_hypotheses",
    "contrasting_action",
    "disambiguating_action",
    "displacement_costs",
    "hypothesis_weights",
    "identify_by_model",
    "identify_roles",
    "model_consistency",
    "persistence_costs",
    "self_log_evidence",
    "self_posterior",
]
