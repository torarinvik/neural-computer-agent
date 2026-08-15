"""Vote the noise out before searching, instead of searching through it.

Two attempts to make induction noise-tolerant failed, and they failed for the
same reason from opposite directions.

Statistical state merging (ALERGIA) compares output proportions with a
Hoeffding bound. At the counts a short-episode prefix tree produces, that bound
is vacuous -- most cells are seen once or not at all -- so everything is
"compatible" and a four-state threshold rule came back as one state at zero
noise.

An exact search with a violation budget lets a disagreement be spent rather
than fatal. But a violation may be spent *anywhere*, so the branching
multiplies and the search stops finishing: the same threshold rule went from
identified to nothing at all.

Both were attempts to make the *search* robust. The evidence is where the
robustness actually lives.

With many short episodes the same short prefix recurs constantly. At a hundred
and twelve episodes over four symbols, every depth-one cell is visited about
a hundred and twelve times and every depth-two cell about twenty-eight. A
label that disagrees with a hundred others is not a contradiction to be
searched around, it is a coin flip to be outvoted.

So: majority-vote each prefix cell, and **drop** the cells too thin to vote on
rather than trusting them. What comes out is a trace with no
noise-induced contradictions in it, which the exact search -- the method that
already works, and the only one that ever identified anything here -- can then
take unchanged. The cost is the deep evidence, which is discarded; the exact
search already treats an ineligible step as constraining transitions but not
outputs, so dropping is a supported operation rather than a hack.

The obvious objection is that this cannot invent information. It cannot. It
converts a problem the search cannot solve into a smaller one it can, and the
measurements in `test_prefix_denoising.py` are about where that trade stops
paying.
"""

from __future__ import annotations

from dataclasses import dataclass

from .identification_ceiling import Trace

DENOISE_SCHEMA = "neural-computer.prefix-denoising.v1"
MIN_COUNT = 3
MARGIN = 0.0


@dataclass(frozen=True)
class DenoiseReport:
    """What voting changed, so a caller can see what it paid."""

    traces: tuple[Trace, ...]
    kept: int
    dropped: int
    corrected: int
    observations: int

    @property
    def kept_fraction(self) -> float:
        return self.kept / self.observations if self.observations else 0.0

    @property
    def correction_rate(self) -> float:
        """How much of the surviving evidence the vote overturned.

        A useful sanity signal: it should track the noise rate. If it is zero
        the evidence was already clean; if it is large the vote is doing
        something drastic and the result deserves suspicion.
        """

        return self.corrected / self.kept if self.kept else 0.0

    def payload(self) -> dict[str, object]:
        return {
            "schema": DENOISE_SCHEMA,
            "kept": self.kept,
            "dropped": self.dropped,
            "corrected": self.corrected,
            "observations": self.observations,
            "kept_fraction": self.kept_fraction,
            "correction_rate": self.correction_rate,
        }


def denoise(
    traces,
    *,
    min_count: int = MIN_COUNT,
    margin: float = MARGIN,
) -> DenoiseReport:
    """Majority-vote labels over recurring prefixes; drop the thin ones.

    `min_count` is how many visits a cell needs before its vote is trusted,
    and `margin` how far from a tie the vote must fall. A cell that fails
    either test is not guessed at -- its steps are marked ineligible, which
    keeps their transition information and discards their labels.
    """

    if min_count < 1:
        raise ValueError("a vote needs at least one observation")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must leave room on both sides of a tie")
    episodes = tuple(traces)
    if not episodes:
        return DenoiseReport((), 0, 0, 0, 0)

    child: dict[tuple[int, int], int] = {}
    ones: dict[tuple[int, int], int] = {}
    totals: dict[tuple[int, int], int] = {}
    nodes = 1
    paths: list[list[tuple[int, int]]] = []
    for trace in episodes:
        node = 0
        walk: list[tuple[int, int]] = []
        for position, symbol in enumerate(trace.symbols):
            key = (node, int(symbol))
            if key not in child:
                child[key] = nodes
                nodes += 1
            walk.append(key)
            if trace.eligible[position]:
                totals[key] = totals.get(key, 0) + 1
                ones[key] = ones.get(key, 0) + int(trace.outputs[position])
            node = child[key]
        paths.append(walk)

    verdicts: dict[tuple[int, int], int | None] = {}
    for key, seen in totals.items():
        if seen < min_count:
            verdicts[key] = None
            continue
        rate = ones.get(key, 0) / seen
        if abs(rate - 0.5) <= margin:
            verdicts[key] = None
            continue
        verdicts[key] = int(rate > 0.5)

    kept = dropped = corrected = observations = 0
    cleaned: list[Trace] = []
    for trace, walk in zip(episodes, paths):
        outputs = list(trace.outputs)
        eligible = list(trace.eligible)
        for position, key in enumerate(walk):
            if not trace.eligible[position]:
                continue
            observations += 1
            verdict = verdicts.get(key)
            if verdict is None:
                eligible[position] = False
                dropped += 1
                continue
            kept += 1
            if outputs[position] != verdict:
                corrected += 1
                outputs[position] = verdict
        cleaned.append(
            Trace(
                symbols=trace.symbols,
                outputs=tuple(outputs),
                eligible=tuple(eligible),
                symbol_count=trace.symbol_count,
            )
        )
    return DenoiseReport(
        traces=tuple(cleaned),
        kept=kept,
        dropped=dropped,
        corrected=corrected,
        observations=observations,
    )


def induce_denoised(
    traces,
    *,
    min_count: int = MIN_COUNT,
    margin: float = MARGIN,
    **kwargs,
):
    """Vote, then hand the cleaned evidence to the exact search."""

    from .identification_ceiling import infer_machine

    report = denoise(traces, min_count=min_count, margin=margin)
    if not report.traces or report.kept == 0:
        return None, report
    return infer_machine(report.traces, **kwargs), report
