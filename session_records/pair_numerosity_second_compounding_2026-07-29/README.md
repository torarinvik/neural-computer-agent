# Second prospective numerosity compounding step — 2026-07-29

## Question

Does the four-lifetime numerosity breakthrough compound again, or was it a
one-time benefit from refining an unusually close boundary?

The frozen parent had been trained at a `23.0%` dot-appearance blend. A fresh
8,192-lifetime curve showed that it generalized through `24.0%`, was unstable
at `24.2–24.6%`, and failed on all three independent seeds at `24.8%`. We
therefore pre-registered `24.8%` as the first clean next frontier.

## Small experimental ladder

All candidates continued the same existing numerosity slot. Controller size
remained 406,456 parameters, with 18,265 existing parameters trainable. All
training used RGB frames, opaque actions, and scalar attempted-action outcomes
on local PyTorch MPS.

1. Four new lifetimes and 16 replay updates missed at `89.93%`.
2. The same four lifetimes with 24 updates passed one seed at `90.22%` but a
   fresh seed failed at `89.51%`.
3. Thirty-two updates did not rescue the hard seed (`89.57%`).
4. A two-stage `24.4%→24.8%` curriculum over eight lifetimes regressed to
   `87.79%`.
5. Eight diverse lifetimes presented together at `24.8%` produced the first
   stable candidate.

This localized the constraint to evidence diversity rather than parameter
capacity, replay depth, or a need for another specialist.

## Replication and shuffled controls

The fixed winning recipe used eight new logical lifetimes, or 48 verifier bits,
and sixteen optimizer updates.

| Seed / condition | Small-screen normal | Counterfactual | Pixel flip | Screen |
|---|---:|---:|---:|---:|
| 24031, real outcomes | **90.29%** | **90.78%** | **81.17%** | pass |
| 24031, shuffled outcomes | 86.14% | 86.74% | 73.42% | fail |
| 24032, real outcomes | **90.33%** | 89.80% | **80.25%** | borderline |
| 24032, shuffled outcomes | 88.87% | 88.31% | 77.88% | fail |

The second real seed missed only the counterfactual cutoff on the 2,048-case
screen. We therefore evaluated both children and the parent on one shared
32,768-lifetime stream rather than selecting by small-sample noise:

| Controller | Normal | Counterfactual | Pixel flip | Accepted |
|---|---:|---:|---:|---:|
| frozen parent | 89.83% | 89.95% | 79.97% | no |
| child 24031 | **90.79%** | **90.90%** | **81.85%** | yes |
| child 24032 | **90.14%** | **90.20%** | **80.56%** | yes |

Thus both real children pass the large shared causal audit, while both matched
shuffled-outcome controls fail.

## Full selected-child audit

Seed 24031 was audited on 8,192 fresh lifetimes per condition:

- `24.8%` target: **90.85%**, versus frozen parent **89.77%**;
- missing-second-object accuracy: **61.43%**, over 29 points below intact;
- inherited `22.4%` numerosity: **+0.79 points** versus parent;
- inherited `23.0%` numerosity: **+0.88 points** versus parent;
- worst magnitude delta: **−1.79 points**;
- worst relation delta: **−0.92 points**;
- worst unrelated-task delta: **−0.93 points**.

Every pre-registered mastery, counterfactual, causal-input, and parent-relative
two-point retention gate passed.

## Conclusion

This is a second prospective within-numerosity compounding step. The sequence
is now:

| Acquisition | New lifetimes | Frontier |
|---|---:|---:|
| magnitude→numerosity bridge | 16 | `22.4%` |
| first same-slot continuation | **4** | `23.0%` |
| second same-slot continuation | **8** | `24.8%` |

The new step requires half the experience of the initial acquisition while
crossing a three-times-larger appearance interval than the first continuation.
It uses no new parameters and retains the registered repertoire. This rules
out the narrow claim that the previous 75% reduction was only a single lucky
continuation.

The result does **not** show monotonically decreasing samples per rung: the
stable floor rose from four to eight as the jump became harder. The next
frontier is therefore adaptive difficulty control—choose the largest
prospective rung that can be replicated under a fixed experience budget—then
test whether accumulated numerosity experience accelerates an adjacent,
previously unseen counting primitive.

## Selected artifact

- `unified_pair_numerosity_second_compounding_seed24031.pt`
- SHA-256:
  `1dcd9149089feaa1c54b3b5a24131716fe39c551cef6fd8ef3b9f67db94b0dc9`

Machine-readable training, control, and full-audit reports are preserved in
[`reports/`](reports/).
