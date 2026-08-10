# Overlapping-mask external routing — 2026-08-10

This is the promoted follow-up to the rejected complementary-mask frontier.
The controller and state adapter remain frozen. The external memory receives
explicit observed-value and observation-mask channels, while the router adds a
generic verified-support prior: a protected cell is penalized when its verified
prototype does not cover the query's observed dimensions.

Copy-on-write also neutralizes value and route weights for dimensions the
source cell never observed. Repeated low outcomes in masked mode quarantine a
protected cell instead of mutating it; a new cell can learn the reversal while
the old capability remains byte-stable. Mastery uses a deterministic held-out
prefix verifier, with those verifier bits recorded separately from training
outcomes.

Both seeds pass the full promotion gate: delayed/noisy scalar credit,
caller-free routing to a new cell, overlapping-mask transfer faster than a
matched fresh learner, reversal, shuffled-outcome/action controls,
missing-evidence no-op, corruption detection, exact reload, protected
retention, frozen controller/adapter, and zero replay.

| seed | successor updates | matched fresh | transfer ratio | unique verifier bits |
| ---: | ---: | ---: | ---: | ---: |
| 85301 | 9 | 23 | 2.5556 | 394 |
| 85302 | 11 | 19 | 1.7273 | 407 |

Reproduce from the repository root:

```bash
.venv/bin/python -m experiments.policy_free_intention_routing.train \
  --seed 85301 \
  --masked-context \
  --mask-curriculum overlapping \
  --report-out /tmp/policy-free-intention-overlap.json
```

This qualifies bounded overlapping-mask transfer and non-destructive masked
reversal. Arbitrary missing-stream reasoning, unrestricted growth, compression,
and general continual learning remain unqualified. The gradual mask schedule
is retained as the next rejected pressure test because the current halfway
switch is still too abrupt for reliable transfer.
