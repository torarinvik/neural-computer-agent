# Temporal binder supervised-bootstrap plan of record

Date: 2026-07-22. This branch is explicitly **supervised-bootstrapped**. It does not establish
reward-only discovery.

## Scale and budget

- Start from the clean spatial/shape controller and consolidator checkpoints, never a pilot.
- Use balanced batch order `temporal, temporal, spatial, shape`, batch size 32, direct-colour
  feedback, and distinct lifetime seeds throughout.
- Each 8,192-lifetime epoch contains 4,096 unique temporal lifetimes and 4,096 rehearsal
  lifetimes. Four epochs contain 32,768 total experiences, 16,384 unique temporal lifetimes, and
  1,024 optimizer updates.
- The joint pilot measured roughly 5 seconds per update before final evaluation. Expected training
  time is about 85 minutes; allow 1.5--2 hours including checkpoint probes and audits. The live
  instance costs $0.3916/hour including storage, so the expected run cost is roughly $0.59--$0.78.

## Checkpoint cadence and 25% gate

Run epoch one as a resumable 256-update phase, then pause. Its expected cost is about 21 minutes of
training plus evaluation/probes, roughly $0.14--$0.18. Resume with the same optimizer and temporary
head state only if:

- gradients and losses are finite and the event-binder gradient is nonzero;
- the residual has moved away from exact zero without exploding above 0.5 RMS;
- a fresh held-out probe is independent of the temporary training head;
- spatial or shape retention has not fallen by more than five points at this early safety gate.

Chance-level rule accuracy at 256 updates is **not** a stop condition: every successful supervised
relative showed a long valley and ignition near or beyond 1,000 updates. A flat early behavioral
curve is likewise not evidence.

After resuming, save every epoch. Fresh held-out probes measure rule decodability at raw write,
compact row, and recall. Larger retention evaluations run at each analysis pause; training-head
accuracy is telemetry only and never counts as a gate.

## Success gates

1. Temporary-head accuracy rises on unique temporal lifetimes (leading optimization indicator).
2. A fresh held-out probe reaches at least 65% at raw write and exceeds shuffled-label chance.
3. Compact row and recall each remain at least 65% and within five points of the preceding stage.
4. Held-out behavioral temporal accuracy reaches at least 65% after demonstrations and improves by
   at least ten points over zero-shot/order-blind behavior.
5. Spatial and shape retention remain within two points of the paired clean baseline in the final,
   high-precision audit.
6. Behavioral reversal, memory-corruption dependence, and original thin-line graduation pass the
   existing integration-ladder gates.

If write decodability rises but collapses during consolidation, lightly fine-tune the consolidator
under rehearsal. If retention shows a reproducible worsening trend, enable the pre-registered L2
penalty on the task-agnostic event-binder residual. Neither intervention is enabled preemptively.

## Phase-one gate result

Phase one completed 256 updates in 23.7 minutes and passed the red-flag gate. Mean binder gradient
norm was 0.307, residual RMS was 0.075, and no numerical failure occurred. Spatial and shape
rehearsal ended near 49% compact retention and 45% end-to-end retention. The temporary head remained
in the expected valley (48.74% mean accuracy, 0.7066 mean loss).

A fresh, independently initialized probe on 1,024 held-out lifetimes reached 55.76% best raw-write,
54.98% compact-row, and 55.76% recalled-vector accuracy. This does not pass the 65% representation
gate, but at update 256 it is a pre-ignition observation rather than a stopping condition. No stage
showed selective attenuation. Training resumed from the exact optimizer and temporary-head state
for epochs two through four.

## Four-epoch result and update-accounting correction

The four-epoch run completed successfully, but did not ignite. The temporary head ended at 49.83%
mean accuracy and 0.6944 loss; held-out temporal behavior remained at chance after demonstrations.
A fresh 2,048-lifetime probe reached only 52.69% best raw-write, 52.05% compact-row, and 51.90%
recalled-vector accuracy.

This is a bounded pre-ignition result, not a failed-bootstrap conclusion. The 1,024 advertised
updates include the full `temporal, temporal, spatial, shape` cycle. Only 512 updates carried the
rule auxiliary loss. That is below the earliest prior direct-colour ignition near 640 signal-bearing
updates and well below the cached-binder 1,000--1,400 update range. Distinct-data scale was adequate
(16,384 temporal lifetimes), but optimization-step scale was not.

The final checkpoint contains model, consolidator, optimizer, temporary head, completed-epoch
counter, and history. The next staged continuation resumes it through epoch eight, reaching 1,024
temporal auxiliary updates. Probe there; if still flat but mechanically healthy, epoch eleven is the
pre-registered upper continuation, reaching 1,408 temporal updates.
