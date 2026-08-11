# Recipe search: expressibility, and what actually cuts the cost

Findings F160-F173. The arc: a misdiagnosed expressibility hole, its
real cause, and three mechanisms measured against each other on search
cost — of which the one that mattered was the one nobody had built.

## What is here

| files | what they measure |
| --- | --- |
| `mod-off-*`, `mod-on-*`, `mod-inf-*` | modulus absent / searched / observed, 4 seeds |
| `m4-*` | the same three arms at 3 seeds, launched together, threads pinned |
| `gated-*` | walled and procedurally gated families as synthesis targets |
| `lib4-*` | reuse with its causal null (F161), 5 seeds |
| `filt2-*` | coverage filter and reuse composed (F171), 5 seeds |
| `enum-*` | depth-ordered enumeration (F173), 4 seeds |

## The results, in the order they were established

**F160.** `toggle` was not inexpressible for the reason recorded twice
in this project and once in a prompt sent to another agent. At two
values per slot a bit flip IS an increment, so `INC i ; INC j`
expresses it. The real hole was that instructions did arithmetic mod
VALUES=8 while each family carries its own value count: `INC 0 ; INC 1`
reproduces toggle's action 0 on exactly 50% of states, per-slot match
[0.5, 0.5, 1, 1, 1, 1].

**F162, F166, F167.** Making the modulus an instruction argument closes
it. Observing the modulus from the data beats searching over it,
+0.0485 against +0.0285 on families whose value range is narrow, over
20 family-seeds and five cleanly paired seeds. Two claims made along the
way did not survive: that observing removes the full-range penalty
(-0.0014 against -0.0040, indistinguishable from nothing) and that it
fails to repair extrapolation (a VACUOUS test — those two arms share
their training, so the comparison could not have failed).

**F169.** End to end: mean held-out 0.9742 against an identity floor of
0.5623, above floor in 21 of 21 family-seeds, interpreter executing
never-seen programs at 0.9916. Against F155's 0.9247/0.5229.

**F170.** The equality guard was NOT built. Gated families reach 0.9607
against 0.9781 for plain ones; the hole the guard would fill is not
there. Measuring cost three runs.

**F161, F171, F173.** Search cost, each against the frozen control:

| mechanism | diverse | related |
| --- | ---: | ---: |
| stored programs, with causal null | 0.929 | 0.772 |
| coverage filter | 0.879 | 0.812 |
| both | 0.848 | 0.711 |
| **depth-ordered enumeration** | **0.425** | **0.406** |
| enumeration + stored programs | 0.421 | 0.376 |

Enumeration is the only one that changes what is COUNTED rather than
which candidates get drawn. `toggle` goes from 22,151 calls to 328.

## What deflated

Stored programs add nearly nothing on top of enumeration (0.421 against
0.425). Reuse was worth about a tenth against random sampling and about
nothing against a systematic proposer. F161's measurement stands, with
its causal null and its direct observation of verbatim recall; its
importance does not survive a better baseline.

## Instruments this arc produced

* **Byte-identical output** across arms that should differ. Caught the
  attainable-fit bound being the coverage filter in disguise, and a
  modulus "fix" that fixed nothing.
* **A positive control of known size.** `cover` is established at
  0.879/0.812, so running it alongside anything new separates "the new
  mechanism does nothing" from "this measurement can see nothing". A
  4000-update smoke read it at 0.994.
* **Thread count pinned in-script.** Two batches launched with
  different OMP_NUM_THREADS train the same seed to different plants,
  which silently broke a comparison described as exactly paired.

## Reproduce

```
PYTHONPATH=src:. .venv/bin/python \
  experiments/games_amodal/probes/isa_compose.py \
  --seed 69316 --train-updates 40000 --weight-decay 0.01 \
  --infer-moduli --synthesize 4000 --library-arms \
  --extra-families 4 --related-families 9 \
  --arms frozen,cover,enum,enum+store
```
