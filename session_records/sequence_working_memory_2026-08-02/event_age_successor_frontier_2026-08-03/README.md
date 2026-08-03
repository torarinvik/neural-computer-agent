# Generic event-age successor slot (pre-registration, 2026-08-03)

## Hypothesis

The successor slot has access to the learned event stream but not an explicit
generic clock. A normalized event-age read may let a fresh zero-output slot
assign scalar-outcome credit to the correct query phase without adding a
task-specific or modality-specific branch. This is a single architectural
input change; no difficulty weighting, suffix window, distractor change, or
privileged label is included.

## Pre-registered arm

- parent: `artifacts/checkpoints/span11_missing_evidence_rehearsal_seed996047.pt`;
- append one 256-wide zero-output successor slot with
  `--skill-adapter-reads-event-age`;
- 128 fresh span-11 mixed-operation target lifetimes;
- protected 128 span-10, 128 span-9, and 128 blank span-11 lifetimes;
- 32 epochs, batch size 512, learning rate 0.0005, binary complement and
  outcome-conditioned critic losses, gate/logit protection 0.1;
- corrected 256-lifetime screen first; scale only after acquisition,
  causal-slot, retention, blank, and reset gates pass.

## Boundary

The event age is a generic stream clock derived from the controller state. It
is not a span ID, operation label, correct action, or semantic coordinate.
The new slot must improve the held-out target over its parent and must be
causally necessary; a positive zeroed-slot gap alone is insufficient.

## Follow-up pre-registration: scalar difficulty on event age

The event-age-only arm was screened separately and did not pass acquisition.
One follow-up is allowed because the event-age route was specifically meant
to support outcome-only credit allocation: retain the same slot and controls,
set `--query-difficulty-power 1.0`, and otherwise change nothing. The weight
is computed only from attempted-action scalar outcomes within generic event-
age buckets; it never reads correct or unattempted actions.
