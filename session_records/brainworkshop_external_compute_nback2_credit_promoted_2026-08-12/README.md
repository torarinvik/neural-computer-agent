# Attempted-outcome credit for external n-back learning

This promotion addresses the hardest failure in the previous open-growth
screen: a fresh external n-back-2 file had enough information to solve the
task, but reinforce training collapsed to the 75% majority-action baseline.

## Change

The external file now supports a generic attempted_bce credit mode. For the
action actually taken, its logit is trained against that action's scalar
verifier outcome. No correct-action label, unattempted-action outcome, task
identifier, or privileged sequence rule enters the learner. A small entropy
term preserves exploration.

The same-cue route reversal and five-file open-growth protocol are otherwise
unchanged. A matched fresh n-back-2 control receives shuffled scalar outcomes
under the same credit rule.

## Results

| Measure | Seed 17 | Seed 18 |
| --- | ---: | ---: |
| Admitted files | 5 / 5 | 5 / 5 |
| Direct n-back-2 accuracy | 1.0000 | 1.0000 |
| Minimum routed-file accuracy | 0.8828 | 1.0000 |
| Shuffled-feedback control maximum | 0.4479 | 0.2760 |
| Same-cue replacement accuracy | 1.0000 | 1.0000 |
| Old-file retention minimum | 1.0000 | 1.0000 |
| Replayed examples | 0 | 0 |

The controller, event encoder, and admitted external files remain frozen
during routing and reversal. Route reload is exact and all promotion gates
pass. The shuffled-feedback control fails mastery, supporting causal use of
the scalar outcomes.

Each seed used 777,728 primary training verifier bits, 73,728 control bits,
22,272 audit bits, 64,128 primary logical lifetimes, 6,144 control logical
lifetimes, 960 primary optimizer updates, 192 control optimizer updates,
1,092 route-memory updates, and zero replay.

## Claim boundary

This promotes a replicated outcome-only credit mechanism that makes a new
external n-back-2 capability learnable while preserving the frozen shared
core and previously admitted files. It does not establish arbitrary
computation, unrestricted growth, consolidation, or general continual
learning. The next pressure test is repeated acquisition of harder working
memory depths and rules, with the same shuffled-feedback and retention
controls.
