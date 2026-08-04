# Consolidated retention and transfer lead

v43 applies the parent-preserving counterfactual protocol to a requested
2,048-update retention phase, with early stopping after three consecutive
held-out validations satisfy retention utility and mastered-parent gates. The
run stops at 320 retention updates rather than selecting a late lucky state.

Seed 19 reaches `1.000` intact recall, `0.520` clear-memory, `0.481` corrupt,
`1.000` reversed, `0.997` target-first, `0.996` target-last, and `1.000`
retention on unseen event tokens. Mastered single-event retention is `1.000`.
The stable threshold is reached at 25,600 unique verifier bits.

The matched transfer curve uses identical unseen event tokens and verifier
worlds for the retained and fresh learners. The retained learner reaches its
stable threshold at 13,312 unique bits; the fresh learner requires 20,480.
This gives a fresh-over-transferred stable-bit ratio of `1.538x`, while the
retained model is already at `1.000` zero-shot intact recall on the unseen
tokens.

This is a one-seed longer-rung transfer lead, not a population promotion. No
checkpoint is promoted. Replicate the transfer ratio across seeds and then
qualify persistent-memory writes, reload, corruption recovery, and transfer
retention before claiming reusable long-horizon capability.
