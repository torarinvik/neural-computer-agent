# Adaptive experience processing

## Question

The verified six-action learner required replay 16 to turn 7,680 unique
contexts into stable behavior. Can the controller decide from its own
prediction error when further processing is no longer useful, preserving
sample efficiency while reducing optimizer updates, replayed examples, and
latency?

Yes. A frozen loss-triggered replay rule replicated on two prospective matched
seeds and passed the complete six-action capability audit.

## Mechanism

Fixed replay performs 16 minibatch updates after every new experience batch.
Adaptive replay treats 16 as a maximum. After each update it measures
Smooth-L1 loss over all experience observed so far and stops processing that
batch once loss is at or below `0.14`.

The gate sees only the controller’s generic latent features and scalar
verifier outcomes already stored in replay. It receives no task identity,
correct action, hidden state, semantic label, or extra verifier outcome.

Both adaptive and fixed arms receive exactly:

- 7,680 unique logical lifetimes;
- 15,360 composition verifier bits;
- the same contexts in the same order;
- the same architecture, initialization, optimizer, and maximum update count.

## Localization

Successful fixed-replay routers ended near loss `0.121`, down from the
zero-predictor loss of `0.320`.

Loss target `0.13` preserved stable mastery but saved only 16.7% of updates on
one prospective seed, missing the pre-registered 25% processing gate. It was
rejected.

The monotonic `0.14` efficiency fork preserved the matched fixed learner’s
exact stable threshold on seed 8105 while saving 36.0% of updates. It was then
frozen before seeds 8106 and 8107.

## Prospective matched replications

| Seed | Stable bits adaptive/fixed | Adaptive updates | Fixed updates | Update saving | Adaptive/fixed wall time |
|---:|---:|---:|---:|---:|---:|
| 8106 | `5,760 / 5,760` | 1,237 | 2,048 | `39.6%` | `2.24s / 3.77s` |
| 8107 | `6,000 / 6,000` | 1,079 | 2,048 | `47.3%` | `1.99s / 4.18s` |

Replayed examples fell:

- seed 8106: 122,880 → 74,220;
- seed 8107: 122,880 → 64,740.

Adaptive final utilities were `0.87954` and `0.87520`, both safely above their
inherited-plus-two-point gates. The stronger fixed final utilities are
recorded honestly (`0.88125`, `0.89637`); the result is processing efficiency,
not higher asymptotic accuracy.

## Independent capability audits

Each adaptive checkpoint was evaluated on eight unseen 2,040-context streams,
a separate 2,400-record randomized confirmation set, retention tasks, and
persistent-memory integrity checks.

| Audit | Seed 8106 router | Seed 8107 router |
|---|---:|---:|
| inherited hierarchy | `0.82710` | `0.82860` |
| adaptive composition | `0.87781` | `0.87328` |
| absolute gain | `+5.07` points | `+4.47` points |
| shuffled router features | `0.73059` | `0.73851` |
| reversed router decision | `0.41547` | `0.42235` |
| randomized lower 95% gain | `+3.58` points | `+2.64` points |
| all eight streams improve | yes | yes |
| binary/four-rule retention | yes | yes |
| exact reload/corruption/parent survival | yes | yes |

The verified descendants of five-action skill `31c77292b2935bf0ce30` are:

- seed 8106: `57d96782a7437350f3b0`;
- seed 8107: `7a1b4e9115135b5561ad`.

Their immutable provenance records the exact adaptive loss target, actual
optimizer updates, actual replayed examples, verifier bits, and complete
ancestral lineage.

## Conclusion

This is the first verified adaptive processing result in the project. The
controller can use its own residual prediction error to decide how much to
learn from fixed experience, preserving the exact stable sample threshold
while cutting processing by roughly 40–47%.

The next frontier is learning the replay stopping criterion itself across
task generations instead of fixing the generic loss target at `0.14`.
