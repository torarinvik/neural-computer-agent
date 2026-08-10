# Distribution shaping: what actually fixes the hard families (F88-F91)

## F88 — "toggle is hard" was the wrong description; SLOT COUNT is the predictor

154 novel in-support families, 2 seeds. Read accuracy falls monotonically with
slot count: 0.996 / 0.995 / 0.995 / 0.974 / 0.935 / 0.870 for slots 1-6.
State-space size is NOT the cause — 512-state families read 0.977, better than
6-slot families at 0.870.

## F89 — entry capacity is not the limit; more of it makes wide families WORSE

Bank tokens 16 -> 48: slots<=3 moves -0.004, slots>=5 moves -0.040, slots=6
moves -0.088. This is F77's result by a different route (F77 widened the
modulation channel, -0.115 on novel read). Rule, now supported twice:

> The bank interface should be as narrow as the task allows. Extra conditioning
> capacity is spent overfitting the training distribution, not on expressing
> harder families.

## F90/F91 — distribution and schema, and what each one does

| slots | rejection+narrow | balanced+narrow | balanced+WIDE |
| ---: | ---: | ---: | ---: |
| 1-3 (mean) | 0.996 | 0.954 | 0.988 |
| 5-6 (mean) | 0.918 | 0.965 | 0.973 |

Balancing slot counts alone TRADES: +0.047 on wide, -0.042 on narrow. Adding
the op vocabulary as well recovers the narrow loss and keeps the wide gain — so
F90's "fixed budget being reallocated" was too strong. A better SCHEMA raises
the ceiling for everyone; distribution shifts within a fixed schema trade off.

## toggle, the project's hardest case since F79

| F79 | F84 (wide@20k) | F87 (narrow@40k) | F91 (balanced+wide@40k) |
| ---: | ---: | ---: | ---: |
| 0.096 | 0.306 | 0.198 | **0.917** |

Zero gradient steps, frozen plant, one forward pass over 128 observed
transitions. Neither ingredient alone was close. The hand-made families are
never sampled by the generator, so this is out-of-set generalisation.

Cost: acquisition 22.2 vs cold 58.9 (2.7x). The rejection-sampled arm's 8.7x is
against an easier test set (cold 51.1).
