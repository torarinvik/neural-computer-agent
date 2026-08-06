# Frozen-core recurrent external transfer — 2026-08-06

Status: rejected as a replicated positive-transfer claim; retain as the
current external-growth diagnostic boundary.

This rung compares a mastered span-2 parent with a cold-start learner on a
new span-4 reverse procedure. The inherited arm receives the parent controller
and trains only an external recurrent growth slot. Both arms receive the same
fresh target episodes, and both receive fresh span-2 parent-task episodes as
online retention rehearsal. The fresh arm is deliberately more permissive:
all of its parameters are trainable. No old examples are replayed.

Seed 69316 is a positive transfer result: the inherited learner reaches stable
target mastery at `6,144` verifier bits versus `9,216` for the fresh learner,
for a `1.50x` fresh-over-transferred ratio. Parent retention is `1.000`, the
frozen-core digest is unchanged, target accuracy is `0.875`, and the
reward-shuffled control is `0.469`.

Seed 69317 passes target, retention, immutability, and shuffled-outcome gates,
but ties the fresh learner at `12,288` stable bits. The population gate is
therefore not met. Width-128, stronger-parent, and learned-output-gate
controls did not turn the tie into a replicated positive gain.
Paired-counterfactual action credit was also rejected as a sufficient repair:
on the failing seed it reduced the transfer ratio to `0.80`, even though its
shuffled-outcome control remained near chance.

The useful result is narrower: recurrent external state plus fresh online
retention outcomes can acquire a new procedure without changing the core or
destroying the parent. The remaining bottleneck is stable sample-efficiency
advantage over a cold-start learner, especially when the inherited parent is
only moderately mastered. This does not establish general continual learning,
unrestricted memory growth, or arbitrary new computation.

Evidence:

- `report_seed69316.json` — positive single-seed transfer lead.
- `report_seed69317.json` — retention-safe tie; no promotion.
- `controls/` — static-slot, no-rehearsal, width, parent-quality, and gated
  controls retained as regression evidence.
