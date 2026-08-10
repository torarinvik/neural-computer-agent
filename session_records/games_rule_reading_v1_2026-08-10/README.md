# The mechanism meets the games battery (F99, F100)

First contact between F71-F98's reading mechanism and the actual games: real
screens from `FamilyVerifier`, verifier-private rules, reward instead of
next-state.

## F99 — the `dual` game: works, and the entry is causally load-bearing

Each trial: avatar centred, `arity` items adjacent, a cue across the top row;
for cue k exactly one side is edible. A family is one (rule0, rule1) pairing.

| arm | choice accuracy | mean reward |
| --- | ---: | ---: |
| trained pairings | 0.667 | +0.600 |
| held-out pairings | 0.345 | +0.214 |
| entry WITHHELD | 0.241 | +0.102 |
| STRANGER entry | 0.083 | -0.100 |
| random plant | 0.065 | -0.042 |
| chance | 0.250 | |

Withholding the entry drops behaviour to chance; a stranger's entry drops it
BELOW chance with negative reward — a wrong rule makes the agent eat the wrong
item on purpose. Stronger causal evidence than anything in the synthetic
families, because being wrong is punished here.

Generalisation to held-out pairings is weak and seed-unstable (0.49/0.53/0.56 on
one seed, 0.10/0.22/0.17 on the other). F78 predicts this: SIX training
pairings, where 64 procedural families produced memorisation and 4096 were
needed for reading. The verifier caps `arity` at 3, so 9 pairings is a hard
ceiling — a fact about the benchmark, not the method.

## F100 — 50 variants: learns nothing, and the probe is why

| arm | lift over floor | beats floor |
| --- | ---: | ---: |
| trained variants | -0.001 | 6.0/12 |
| held-out variants | +0.001 | 5.5/12 |
| random plant (null) | +0.003 | 8/12 |

The random plant does as well as 12000 updates of training, in distribution as
much as out. No lift anywhere = formulation failure, not generalisation failure.

Cause: collect/intercept/avoid/navigate are MULTI-STEP, and this probe predicts
the outcome of ONE action with behaviour as greedy argmax. Nearly every single
action yields zero, so the model is right and useless. `dual` worked because a
dual trial IS one step.

F67 prescribed deriving behaviour by SEARCH in a learned transition model, which
`reacher_ladder.py` does. This probe learned a reward model and no transition
model, then searched to depth one. The missing piece is the multi-step
derivation, not the bank.
