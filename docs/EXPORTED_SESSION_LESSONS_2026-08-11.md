# Extracted lessons from the games continual-learning session

Source: `/Users/torarinvikbjarko/Downloads/session-export-1786446043922`.

## Evidence status

All quantitative results and causal conclusions reported by the exported
session are **SINGLE-SOURCE, UNREPLICATED** until an equivalent experiment has
been rerun in this repository with the required controls. This document is a
source-provenance record and hypothesis catalogue, not a promotion ledger.
Repository experiments may validate a mechanism inspired by the export, but
they do not retroactively validate the export's numbers.

This is a distilled research record, not a promotion report. The session
worked in a games-focused clone of the project and repeatedly compared a
frozen plant against external task entries. Its most useful conclusions for
the canonical amodal architecture are below.

## What transferred as hypotheses

The following are architectural hypotheses and design lessons extracted from
the export. Their exported empirical support remains **SINGLE-SOURCE,
UNREPLICATED**.

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

## Export-reported positive result — SINGLE-SOURCE, UNREPLICATED

The games session reported a more important storage rule than ordinary
consolidation: store a factual transition model and derive actions by search;
do not store a task policy. Policies are preferential and become contradictory
when the task changes. A transition model is factual and can be incomplete
without being wrong. In the session's best replicated model-based battery, a
frozen plant retained all earlier entries exactly, novel families were often
read with zero gradient steps, and acquisition cost fell across later tasks.

That result initially looked like general transfer, but follow-up controls
corrected the claim. The first battery mostly used nested dynamics. On genuinely
different families, a flat per-slot model did not learn reusable structure; a
top-down shared transition basis was required. The durable lesson is therefore
not “a model automatically transfers,” but:

`shared transition kinds in the plant + task-specific residual facts in the bank`

This is the best foundation for our controller-as-CPU analogy. The bank may
grow, while the controller executes a stable vocabulary of operations and the
planner derives behavior from facts rather than accumulating stale habits.

## Export-reported computation principle — SINGLE-SOURCE, UNREPLICATED

Three findings within the export support one rule; they are not independent
replications across this repository:

> Do context-dependent work once, then iterate something fixed.

- Model-based planning stores facts once and searches them, rather than storing
  a policy that must be unlearned.
- A shared per-primitive step function composes unseen programs and even
  generalizes to depths never shown during training; a one-shot composite
  interface memorizes seen combinations and fails on new ones.
- A context entry must be bound once into executable parameters before an
  iterative loop. Re-attending or re-decoding the entry at every step caused
  errors to compound; bind-once execution reached near-perfect performance on
  held-out multi-step programs with a correct entry.

This is directly relevant to the canonical runtime: event evidence should be
formed and addressed once, then the controller should perform repeated opaque
think/act steps against a stable bound view. Repeatedly re-solving identity or
re-reading a noisy key inside the loop is a likely source of long-horizon
failure.

## Addressing and memory necessity

The session also isolated a hard chicken-and-egg boundary. Visually identical
contexts cannot be addressed before the agent acts; the correct sequence is
`act -> observe consequence -> address -> fetch -> execute`. Staging these
pieces independently failed because the staged policies were different from
the final probe-conditioned policy. Co-training the probe, address table, and
read path was the first configuration to close the loop, but it exposed two
shortcuts:

- the recurrent state could retain the answer, making the fetched fragment
  decorative;
- a default context could absorb one twin, so entropy/KL penalties only
  suppressed confidence without removing the default behavior.

Therefore a memory claim needs a causal necessity gate: correct entry,
wrong-entry, same-norm decoy, zero-read, and corrupted-entry conditions must
separate. A high score alone is not evidence that the bank carried the skill.
Context diversity or randomized context-to-behavior bindings are more
promising ways to prevent default absorption than simply increasing an
ignorance weight.

## Export-reported failures and corrections — SINGLE-SOURCE, UNREPLICATED

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
- Protected plasticity and EWC demonstrated replay-free retention in one
  fully plastic core, but they are consolidation mechanisms, not the final
  storage solution. Freezing or adding per-task adapters merely relocates
  forgetting into whichever component remains plastic.
- Composition mechanisms trained only on seen pairings did not produce
  zero-shot unseen pairings. Generalization requires a training protocol that
  makes held-out recombination necessary, or an inductive bias that applies
  one primitive at a time.

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
7. Treat reader quality and plant executability as separate gates. Oracle
   entries can establish whether the plant can execute; own-versus-stranger
   entries establish whether the reader is causal. Do not use either result as
   evidence for the other.
8. Record learning curves, not only endpoints. The session's reader appeared
   to have a mechanism-level amortization gap until longer matched training
   closed a large part of it. At the end of the export, curve instrumentation
   had just been added; its 150k/200k saturation results were still pending.
9. Correct planning arithmetic before increasing search. If a value head
   already contains an H-step return, adding it at every beam depth counts
   rewards multiple times. Use immediate rewards along the rollout and one
   terminal value, with each reward weighted once.

## Current implication for this repository

The canonical repo independently has the bind-once operator ABI and durable external
file payloads. The new read-adapter diagnostic remains rejected because the
current generated-composition target can solve without the file. The export
supports the same next move, with one refinement: the target must require both
an external fact and a fixed iterative execution path. The highest-ROI test is
a minimal opaque transition family in which:

1. the controller is frozen;
2. the independently persisted file supplies the task-specific residual;
3. the residual is bound once before a multi-step rollout;
4. a zero-read, wrong-file, corrupted-file, and fresh-learner control are run;
5. the report includes the full held-out learning curve and retention.

This should be treated as a capability-boundary experiment, not yet a claim of
general continual learning. The export's strongest result is therefore a
design guide, not a validated result in this repository:
fixed computation plus growing factual external state is promising, but only
when causal memory use, cross-family structure, and long-horizon execution are
all measured separately.

The matched recency-plus-latest diagnostic is archived in
`session_records/factored_residual_sequence_recency_latest_diagnostic_2026-08-11/`.
It produced the same outcome as the compatibility last-token key on seeds
91–93: 8/9 regime promotions, 0/3 complete gates, and 0/3 missing-evidence
recovery passes. The useful negative result is that the current failure is
not fixed by a simple aggregation change. Cumulative evidence increases
recursive factual-model error until the safe read-only router refuses. The
next target is a horizon-aware verifier or bound-once transition model, with
contradiction refusal intact.

That bound-once execution seam is now implemented as
`ExternalBoundTransitionModel`. Exact transition memory can expose hit
evidence to the planner, and `require_known=True` makes missing rows fail
closed during recursive search instead of allowing a default zero prediction
to affect beam ranking. This is an integrity improvement and a stronger
experimental control; it is not itself a learned capability gain.
