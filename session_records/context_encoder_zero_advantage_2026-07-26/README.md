# Context encoder: zero-advantage bounded negative

Date: 2026-07-26

## Question

Can a tiny learned context metric improve strategy retrieval using only scalar
verified improvement, without task identities, utility labels, or privileged
state?

## Design

- 13 learned positive feature scales
- fixed four-slot strategy bank
- identical fourth retrieval probe in learned and frozen arms
- two paired seeds
- two rounds per phase and four physical banks
- dynamic RAM admission and eviction disabled

The selected retrieval probability was reinforced by its physical verified
reward advantage over the center candidate. The tensor arena remained a parity
audit only.

## Result

Rejected at the sub-minute gate.

For seeds 7060 and 7061, learned and frozen arms were behaviorally identical.
All ten eligible updates had zero verified advantage, mean context loss was
zero, and all 13 feature scales stayed exactly 1.0. The model retained the old
binary and four-rule skills.

A shuffled-key intervention on seed 7060 reduced reliability-target choice
from the intact arm's mean 87.5% to 50%, indicating that correct addressing can
matter. It does not establish learned addressing because the intact encoder
received no differentiating verifier signal.

## Conclusion

Do not increase duration or encoder capacity. Hard retrieval followed by an
often behaviorally equivalent candidate creates no useful addressing credit.
The next minimal fork should expose a soft mixture of stored strategies to the
verifier, while keeping the same 13-parameter metric, matched candidate budget,
two-seed replication, shuffled-key audit, and retention gates.
