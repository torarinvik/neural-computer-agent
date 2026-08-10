# Density beats the untrained control — but the twin control shows the bank is unread (F104)

## Density, not updates, is the decisive variable

| arm | held-out lift | vs untrained | wins |
| --- | ---: | ---: | ---: |
| 8k updates, seek 0.5 | +0.0007 | -0.0018 | 4/12 |
| 40k updates, seek 0.5 | -0.0017 | -0.0047 | 2/12 |
| 40k updates, seek 0.85 | +0.0093 | +0.0161 | 6/12 |
| untrained control | +0.0075 | — | — |

40k at seek 0.5 is WORSE than 8k at seek 0.5: more training on a signal-poor
distribution actively hurts. With both, seed 69316 reaches +0.0236 vs the
untrained +0.0075, with large wins on `intercept` variants. Seed 69317 fails.

## The inverted-twin control settles it

Same components, same rendering, opposite rewards — the only actively WRONG
entry. F103's stranger control drew a random variant, usually merely
uninformative.

| entry | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| correct | +0.0236 | -0.0051 |
| withheld | +0.0202 | -0.0063 |
| stranger | +0.0215 | -0.0054 |
| inverted TWIN | +0.0225 | -0.0057 |
| correct − twin | +0.0011 | +0.0007 |

Handing the agent the exact opposite of the truth changes nothing. Most
per-variant differences are exactly 0.0000. The entry carries none of the
inversion bit; the gain is inversion-INVARIANT competence from the transition
model and search.

## The contrast with F99 is the finding

On `dual`, a stranger's entry drove reward to -0.100 against +0.600 correct.
Same mechanism, same reader. The difference:

- `dual`: every step resolves a trial answerable ONLY from prior outcomes, so no
  inversion-invariant policy exists;
- multi-step variants: most available reward is inversion-invariant, so a policy
  that ignores the bank captures it, and gradient descent finds that first.

**A benchmark only exercises context-reading if ignoring context is
unprofitable.** Test-design rule: before adding a game to test the bank, verify
that an inversion-invariant policy scores near floor. If it does not, the game
measures navigation competence and will report the bank as working or failing
for reasons unrelated to the bank.
