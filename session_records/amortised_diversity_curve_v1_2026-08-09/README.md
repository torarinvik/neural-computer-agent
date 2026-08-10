# Training diversity is the knob (F77-F79)

Holding architecture fixed and varying only the number of distinct families
the plant and encoder train on.

    amortised_bank.py --pool {64,256,1024,4096} --train-updates 20000 \
                      --query 256 --context 128

| pool | in-dist | novel read (0 steps) | mastered | acquisition cost | cold cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.974 | 0.318 | 0/16 | 186.8 | 46.1 |
| 256 | 0.919 | 0.682 | 2.3/16 | 83.8 | 49.5 |
| 1024 | 0.931 | 0.918 | 5.5/16 | 49.1 | 45.5 |
| 4096 | 0.968 | 0.907 | 5.5/16 | **34.3** | 50.0 |

Pool 64 has the HIGHEST in-distribution accuracy and the WORST novel accuracy:
it memorised its families instead of learning to read one. Chan et al.'s axis,
measured directly.

**Per-task gate crossed at pool 4096**: 34.3 vs 50.0, both seeds individually
(22.9/41.7 and 45.8/58.3). Every plant weight frozen; retention delta 0.0000;
wrong-context null 0.000-0.138.

**Lifetime gate NOT met**: 20000 pre-training updates against a 15.7-update
per-task saving needs ~1274 downstream families to break even. Sixteen measured.

## F77: the recorded prediction was wrong

Widening the entry channel (per-family gains/biases modulating every block)
made novel-family reading WORSE (0.682 -> 0.567, mastered 2.3 -> 1.0/16) while
in-distribution was unchanged — overfitting the conditioning pathway, not a
capacity limit lifted. The recorded falsifier ("if it degrades retention, any
expressive channel interferes") did NOT fire: retention stayed 0.0000. The wall
is reader generalisation, not interference.

## Confound caught

Two arms at dim128/L3 scored 0.392 and 0.375 in-distribution vs the working
config's 0.918 — undertrained at 20000 updates, not evidence about pool size.
Comparing them would have "shown" that a larger pool hurts.
