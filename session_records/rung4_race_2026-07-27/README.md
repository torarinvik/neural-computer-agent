# Fourth primitive and the growth of the transfer advantage

The third rung established that retained skills make a later skill cheaper to
learn. The obvious next question is whether that discount grows as the library
grows. It does not.

Everything here ran on two RTX 5090s (torch 2.12.0+cu130). Runs were confirmed
bit-deterministic on this device before any measurement was trusted.

## The estimator had to be replaced first

The third rung's committed number came from a gate threshold: the first budget
whose gates pass and whose every later budget also passes. Reproducing the
committed three-seed curve on CUDA moved those thresholds by up to twelve
updates. Since the runs are bit-deterministic, moving from mps to CUDA changes
nothing but floating-point rounding, so that estimator is knife-edge sensitive
to rounding alone. Re-measuring the same rung on eight seeds confirms it: the
committed estimator puts the baseline ahead on two of the eight.

The replacement estimator interpolates each seed's held-out accuracy curve to a
fixed target and pairs within seed, backed by a paired sign test on accuracy at
every budget. It is monotone in the underlying quantity and does not depend on
where a pass/fail boundary happens to fall.

Under the better estimator the third rung's effect is **larger** than reported,
not smaller: median experience ratio 1.231 with all eight seeds above one, and
a pooled accuracy advantage of +4.81 points over 104 paired cells
(sign test p = 1.5e-8). The finding survived; the metric did not.

## The fourth rung

`contextual_composition` is `identity ^ rule ^ context`: the original hidden
binary mapping composed with the acquired visible-context bit. It reuses both
earlier primitives on every event.

Earlier rungs had already claimed both residual slots — the second rung took
the action adapter, the third the relation adapter — so this rung appends a
slot from the new indexable `skill_adapters` stack. The stack is empty by
default and contributes no state, so every checkpoint promoted before it
existed still loads strictly. The inherited controller, both legacy adapters
included, stays frozen and bit-identical; 14,105 plastic parameters sit against
325,139 frozen ones.

Two renderings had to be separated before any of this was measurable:
`visible_context_xor` and `contextual_composition` previously drew the *same*
operation cue, so they rendered bit-identical frames while disagreeing on half
their events. Each operation now owns a cue slot in a band no stimulus glyph or
context marker can reach.

## Result: the advantage shrank

Eight paired seeds per rung, arms differing by exactly one rung of ancestry.

| | third rung | fourth rung |
|---|---:|---:|
| experience ratio at 85% | **1.231** (8/8 seeds, p = 0.0078) | **1.083** (8/8 seeds, p = 0.0078) |
| experience ratio at 90% | 1.192 (7/8, p = 0.070) | 1.116 (6/8, p = 0.289) |
| experience ratio at 95% | — | 1.000 (4/8, p = 1.000) |
| pooled accuracy advantage | +4.81 points (p = 1.5e-8) | +1.16 points (p = 0.105) |

The fourth rung keeps a small, consistently signed advantage at the lowest
target — every seed is above one — but it is about a third the size of the
third rung's and it disappears entirely by the 95% target. There is no
evidence that the discount compounds.

Two caveats keep this from being a clean statement about rung depth alone. The
two rungs are different task families (a zero-shot visible composition against
a hidden-rule few-shot one) and different budget scales (tens of updates against
hundreds). And this rung is mastered at **two** support outcomes; at one support
it reaches only 0.569 in 2048 updates, so its one-support form is unreached.
`binary_mapping` was originally acquired through the same graduated reduction of
support, so that remains the open path rather than a failure.

## Retention is the binding constraint

The more interesting result is where the richer ancestry costs something. Among
the 32 runs per arm at budgets of 768 or more, the three-skill arm failed the
`visible_context` retention gate **14** times against the two-skill arm's
**1**, and its mean retention on that skill was lower (0.911 against 0.935).

The new composition's nearest neighbour is the direct-context skill, separated
by a single cue bar. Each rung adds more to protect in the region the next
skill occupies, and that burden grows faster than the saving on the new skill.
Learning speed is not what limits this ladder; interference is.

## Promoted controller

`artifacts/checkpoints/unified_four_skill_composition_seed8416.pt`, seed 8416,
1536 updates.

- new composition: 99.57% on 2,048 held-out lifetimes, all ten sub-gates pass,
  including zero-shot near chance and shuffled-feedback-hurts, so the rule is
  inferred from the outcome rather than memorised
- retention: binary mapping 97.26%, direct context 91.64%, context XOR 91.80%
- removing only the operation cue drops the new skill to 55.34%
- inherited weights bit-identical; independent reload passes all four skills
- shuffled retention teacher rejected, with all three retentions collapsing to
  between 0.4954 and 0.6090

## Repository fixes this rung required

- `model.py` compared a strided view against a contiguous tensor in a
  bit-identity assertion. It held on mps and failed on this box's CPU kernels
  by 3e-8. The leading feature slice is now materialised so adding a
  zero-initialised statistic is exactly, not approximately, behaviour-preserving.
- `train.py` asserted a second-support rollout existed for every contextual
  task, which no one-support contextual rung renders.
- `train.py` computed per-context accuracy only inside that same
  two-support-only block, so the `both_contexts_mastered` gate crashed at one
  support.
- `environment.py` gave two different operations the same cue, described above.
