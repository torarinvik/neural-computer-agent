# Strategy seed factorization and racing correction

Date: 2026-07-26

## Factorization

The physical experience, two-parameter policy perturbation, and 13-parameter
context-proposal random streams can now be overridden independently.

Using weak seed 7072 as the base:

- experience 7073 alone: zero informative pairs;
- policy 7073 alone: 15.1% informative pairs, zero old return;
- context 7073 alone: 5.7%, zero old return;
- experience + policy 7073: 15.1%, no return advantage;
- experience + context 7073: 9.4%, zero old return;
- policy + context 7073: 22.6%, failed the gate, zero old return;
- all three 7073: 56.6%, 95.8% old-return target accuracy.

The all-three override matched the original strong training trace bit-for-bit.
The large gain is a nonlinear trajectory interaction.

## Exploration race

On physical stream 7072, an exploration clone selected after a cheap screen
later reached 52.8% informative comparisons and 61.1% reliability target
accuracy, versus 17.0% and 1.4% for the original exploration stream. This
shows useful compute-for-experience variation exists.

However, the original screen shortened each phase and therefore was not an
exact prefix. It cannot validate successive halving.

On blind physical stream 7074, the nominal screen winner later produced 24.5%
informative pairs but zero old-return accuracy. A tied diagnostic clone
produced 62.3% informative pairs yet underperformed its frozen comparator and
failed the gate. High information production can coexist with harmful
adaptation.

## Exact-prefix repair

`--max-physical-rounds` now stops the full schedule without altering its phase
length or random streams. Prefix reports are explicitly non-graduating.

A 12-round prefix for clone 7081 matched the first 12 full-run rows bit-for-bit.
But this prefix contains only the old-equal phase. It cannot evaluate phase
switching or retained return.

## Verdict

Seed variance is interaction-driven. Value diversity remains promoted, but
early information density alone is rejected as a clone-selection objective.
The next race must expose all three behaviors—initial utility, reliability
switching, and old-utility return—before selection. The efficient candidate is
a pre-registered interleaved curriculum, compared against the existing blocked
schedule at equal verifier bits.
