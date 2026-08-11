# Extracted lessons from the games continual-learning session

Source: `/Users/torarinvikbjarko/Downloads/session-export-1786446043922`.

This is a distilled research record, not a promotion report. The session
worked in a games-focused clone of the project and repeatedly compared a
frozen plant against external task entries. Its most useful conclusions for
the canonical amodal architecture are below.

## What transferred

- The right abstraction is a fixed controller/plant that consumes an external
  entry, not one controller or one policy per game. A keypress decoder is only
  an output adapter; it should not change the controller.
- Facts and transitions are safer memory contents than policies. When a world
  rule reverses, stored behavior becomes stale, while observations and their
  verified consequences can be reinterpreted.
- A causal external-memory claim requires an own-entry versus wrong-entry
  comparison, not merely a good score. The session's strongest result was a
  frozen plant whose one-pass entry improved held-out unseen worlds and whose
  wrong entry degraded performance.
- Diversity was more valuable than raw capacity. Increasing the number of
  distinct pretraining families made the plant rely on the external entry;
  larger conditioning paths and more capacity often did not help and
  sometimes hurt.
- An ignorance objective was useful: penalizing a plant that succeeds without
  the entry broke the shortcut in which the plant learned an average of a
  world and its inverted twin. Oracle substitution was the fastest way to
  localize whether the reader, plant, or search was limiting performance.
- Small mathematical proving grounds were highly valuable. A twin-rule
  arithmetic task reproduced the games reader's polarity question in minutes,
  making it possible to separate reader capacity from sparse game data.

## What failed or was corrected

- Semi-amortized entry refinement and a discrete codebook were nulls at
  matched two-seed budgets. They should not be treated as improvements merely
  because the literature suggests them.
- The reader improved substantially with longer training, so an apparent
  amortization/mechanism gap was partly undertraining. Final scores alone had
  hidden the learning curve.
- One codebook result was initially misread because all worlds used one code;
  instrumentation exposed codebook collapse before it became a false finding.
- A second value channel did not solve inverted game worlds because the data
  collection policy rarely visited the object type that needed to be learned.
  The missing evidence was a data problem, not a missing architectural slot.
- The games beam scorer double-counted rewards already included in multi-step
  value estimates. The correct planning decomposition is immediate rewards
  along the rollout plus one terminal value, with each reward weighted once.
- More state slots and more search depth were not automatically useful. A
  change needs a predicted numeric signature and a cheap check against
  existing runs before a long training campaign.

## Architectural rules to carry forward

1. Keep learned content outside frozen controller, encoder, and decoder
   weights. External memory may grow, be versioned, compressed later, and be
   replaced independently.
2. Store reusable facts or computational fragments, not opaque per-game
   policies. Require a composition test over unseen arrangements before
   claiming compounding transfer.
3. Make external reads observable and causally necessary. A file that is
   calibrated, persisted, and routed but can be bypassed is infrastructure,
   not learned capability.
4. Prefer one-time addressing followed by fixed recurrent execution. Bind-once
   routing avoids repeated contextual lookup and gives a clean memory ABI.
5. Every promoted rung should report a held-out curve, own/wrong-entry
   controls, shuffled and missing-evidence controls, read or memory ablation,
   exact reload, fresh transfer, retention, and equal-compute accounting.
6. Do not infer a mechanism from one score. First test for undertraining,
   data starvation, collapse, hidden privileged information, arithmetic bugs,
   and mismatched compute.

## Current implication for this repository

The canonical repo already has the bind-once operator ABI and durable external
file payloads. The new read-adapter diagnostic remains rejected because the
current generated-composition target can solve without the file. The next
high-ROI pressure test is therefore not a larger operator bank. It is a
minimal target in which the only route to the held-out computation is an
independently persisted file value, followed by a matched zero-read control
and a fresh-learner transfer curve.
