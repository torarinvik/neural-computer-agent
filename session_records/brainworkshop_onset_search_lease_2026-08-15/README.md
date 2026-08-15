# Onset search lease, 48 steps (2026-08-15)

Status: **rejected**.

Unused seeds 125017, 126017, and 127017 ran the closed-grammar search on the
public onset rule (press when the current symbol is the target **and** it just
changed). Search selected `and(invert(slot 0), acquired prototype)` on every
seed and held it frozen at `1.000` for six sessions, but one pre-registered
single-family control crossed threshold, so the campaign is rejected.
`AgentBrain.bank` was not written and nothing was admitted.

## Results

Winner is `and:0` on every seed, bound to frontend `1ce405a0…`. Controller
digest `59c9ef2b…`. Slot 0 stayed `90e20193…`. 47 eligible trials per session.

| Seed | Acquire | Six frozen holds | Stable prefix | retrieve slot 0 | invert slot 0 | prototype only | zeros | reversed | reward shuffled | other encoder |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 125017 | 0.745 | 1.000 × 6 (282 bits) | 47 | 0.255 | 0.745 | **0.787** | 0.745 | 0.000 | 0.468 | 0.745 |
| 126017 | 0.745 | 1.000 × 6 (282 bits) | 47 | 0.255 | 0.745 | **0.702** | 0.745 | 0.000 | 0.532 | 0.745 |
| 127017 | 0.745 | 1.000 × 6 (282 bits) | 47 | 0.255 | 0.745 | **0.830** | 0.745 | 0.000 | 0.553 | 0.745 |

Seed 127017 fails: a current-symbol prototype alone reached `0.830`, above the
`0.8` threshold, so this population cannot claim that onset needs two families.

## Why it failed

The prototype-only arm presses on every target frame, including repeats, so its
accuracy is pinned to the rule's base rate, near `0.75`. With 47 eligible trials
the sampling spread around that base rate reaches past `0.8`, which is the same
threshold the winner must clear. The 48-step configuration therefore cannot
separate a one-family near-miss from a two-family solution. The control was not
relaxed; episode length was retested on a fresh seed block in
`brainworkshop_onset_search_lease_long_2026-08-15`, where every single-family
control stays at or below `0.759`.

The invert-only arm is at the same base-rate ceiling (`0.745`), and the zeros
and cross-encoder arms fall back to it because an AND with no usable prototype
degenerates to its invert parent.

## Grammar coverage

Of 19 proposals, 5 executed, 7 were recorded illegal, and 4 (retrieve slot 2,
`compose(0,0)`, `compose(1,1)`, invert slot 2) could not be installed: the
prototype-capable machine does not load recursive depth-2 files. Search
therefore covered retrieve {0,1}, invert {0,1}, `and`, and invent on this
machine, not the depth-2 files.

## Limits

- rejected; nothing admitted and no AND slot in `AgentBrain.bank`;
- not a bits-to-threshold transfer ratio against a fresh learner;
- depth-2 files were proposed but not executable on this machine;
- the AND prototype is bound to `1ce405a0…` and does not survive a frontend
  swap.

## Superseded

`brainworkshop_onset_lease_discriminating_2026-08-15` is the standing onset
result: a fresh block (134017-136017) at 447 eligible trials, pre-registered,
winner `and` at `1.000` on every seed with every control below `0.8`. This
campaign stays rejected, and it is the reason the trial floor exists.
