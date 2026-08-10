# The bank stores rules, not exceptions (F95-F96)

F92 localised the mechanism's ceiling to position-dependent dynamics: the
reacher's walled grid costs 438 updates against a cold 88. The ledger called for
a conditional/masked op primitive, by analogy with the fix that took `toggle`
from 0.096 to 0.917.

## F95 — the primitive made everything worse at fixed budget

Adding `wall` (a barrier refusing a move onto one value) and `cond` (an effect
conditional on another slot) at 40000 updates: walled 0.894 -> 0.795, novel read
0.976 -> 0.960, in-distribution 0.979 -> 0.961. The target case got worse.

## F96 — doubling the budget repairs everything EXCEPT the target

| family | bal+wide @40k | gated @40k | gated @80k |
| --- | ---: | ---: | ---: |
| grid | 1.000 ft 0 | 0.978 ft 25 | 1.000 ft 0 |
| toggle | 0.917 ft 125 | 0.800 ft 25 | 0.992 ft 0 |
| dial | 0.782 ft 150 | 0.833 ft 88 | 0.980 ft 25 |
| walled | 0.894 ft 438 | 0.795 ft 600 | 0.894 ft 600 |
| novel read | 0.976 | 0.960 | 0.982 |
| acq vs cold | 2.4x | 1.3x | 3.0x |

## 0.894 is an exact identification, not a partial success

`grid` and `walled` agree on 229/256 transitions = 0.8945. Measured read
accuracy is 0.894 on every seed at every budget. The reader gets every non-wall
transition right and every wall transition wrong — it reads "8x8 grid movement"
and ignores the exception set entirely, despite ~13 of its 128 observed
transitions demonstrating it.

The obstacle is ~121 bits of ARBITRARY content (log2 C(256,27)). "Increment slot
2 mod 8" is a rule; "these 27 cells are blocked" is a list. An entry read by a
plant applying uniform functions of slot values can only name a rule.

## Conclusion: the semantic/episodic split, from measurement

- RULES: compressible, universal — the bank entries this mechanism has.
  0.982 read, 3.0x cheaper acquisition, retention exact to N=1024.
- EXCEPTIONS: arbitrary, per-state — need a STORE, not a rule.
  `ContentAddressedMemory` has been unused in this repo since the start
  (open weakness 8). Prediction: `walled` goes 0.894 -> ~1.000 by storing 27
  exceptions, rule-bank untouched.
