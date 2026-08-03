# Continuation and plasticity-gate frontier — 2026-08-03

These small experiments tested whether the promoted 1,024-lifetime
complement child could improve from a small fresh packet without disturbing
earlier spans. Every claim below is based on an independent 1,024-lifetime
audit; private 256-lifetime screens are reported only as diagnostics.

## Existing-slot continuation

The parent was `complement_population_winner_seed93789.pt` (three frozen
successor slots; full complement audits are typically about 67%). The first
256-lifetime continuation modified its final slot with span-nine/span-ten
rehearsal. It fell from **67.44% to 59.13%** complement accuracy, although old
spans rose slightly. This is a negative for simply continuing a learned slot
on a small packet.

Three same-stream continuation clones (256 fresh lifetimes each) looked
promising on private audits: each was about **+2.0–2.34 points** above its
matched parent. The best clone was then selected and audited on a larger
independent set. It fell from **67.68% to 62.04%**; averaging all three clone
checkpoints was worse at **61.00%**. The shuffled continuation was **51.19%**,
so the update still used reward correspondence, but it did not generalize.
This is a direct warning that a small private screen can select an
overfitted arm. Population selection is not valid evidence of compounding
unless the larger audit also beats the parent.

A gentler 512-lifetime continuation (lower learning rate, fewer epochs,
margin objective, and span-nine/span-ten rehearsal) preserved old skills but
changed complement by **−0.19 points** (67.22% → 67.03%). It is a bounded
plateau, not a breakthrough.

## Appending a new slot

Appending a fourth slot with 256 fresh lifetimes did not help: complement fell
from **67.24% to 64.84%**, while zeroing the new slot returned exactly to the
parent. The new slot was therefore harmful rather than a compounding gain.

## Entropy-only context gate

The code now supports a generic normalized parent-action entropy scalar as a
successor input. Unlike the earlier raw-logit/probability experiments, it
does not expose action-coordinate identity. The interface is checkpoint-
compatible and unit-tested, but the first 256-lifetime append experiment was
strongly negative: **67.35% → 55.96%**, with the zeroed-slot control returning
to the parent. It is archived as a rejected diagnostic, not an active
architecture recommendation.

## Decision

The evidence now separates three effects that were previously conflated:

1. A population can make initial acquisition more reliable, but a noisy small
   private screen can select an overfit continuation.
2. Updating an already learned slot with only 256–512 new lifetimes does not
   yet produce a verified second compounding gain.
3. Appending another residual without a stronger context/plasticity mechanism
   can reduce behavior even when the old slots remain intact.

The next experiment should therefore improve **audit reliability and
promotion criteria** before adding more gate features: use multiple
lifetime-disjoint private seeds (or a larger private sample) and require that
the selected child beats its parent on the aggregate, not only that it beats a
zeroed-slot control. The full causal, shuffled, blank, reset, and old-span
retention gates remain mandatory. Do not promote any checkpoint in this
directory.

The replay trainer now rejects feature-width mismatches explicitly; a replay
buffer collected without a newly introduced successor read cannot be silently
mixed with that interface.
