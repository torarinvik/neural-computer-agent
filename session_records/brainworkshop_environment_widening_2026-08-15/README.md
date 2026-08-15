# Which results were about the agent, and which about the room (2026-08-15)

Status: **diagnostic**, on the already-consumed development seed. Nothing
admitted; `AgentBrain.bank` unchanged at `07319eb1`.

Every result in this session ran in one environment: four positions on a grid,
rendered pixel-identically every time they appear, over a single visual stream.
That is a generous world. The largest distance between two observations of the
same symbol was **0.001**; the smallest between different symbols was **4.64**.
Every stage downstream inherited that separation for free.

So the question is not whether the agent can do more. It is which of the
numbers were about the agent and which were about the room. Two axes move; the
agent does not.

## Result

| alphabet | noise 0.00 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
| ---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 4 | 12/12 | 12/12 | 12/12 | 12/12 | refused | refused | refused |
| 6 | 12/12 | 12/12 | 12/12 | 12/12 | refused | refused | refused |
| 8 | 12/12 | 12/12 | 12/12 | 12/12 | refused | refused | refused |

Acquisition ratio against the matched control, at every solved cell: **0.583**
at four symbols, **0.567** at six, **0.694** at eight. Zero false recognitions
anywhere.

**Alphabet size was not the fragile axis.** Doubling it costs nothing: every
task still solved, accumulation still paying, and the ratio at eight symbols is
within 0.13 of the ratio at four. The machinery is not tuned to four.

**Stimulus noise was the fragile axis, and it broke in one specific place.**

## The single point of failure

Not accuracy. The *alphabet discovery*.

`cluster_events` groups observations that lie within a fixed tolerance of 0.5.
At a noise level of 0.05 the within-stimulus spread passes 0.5, every symbol
shatters into several clusters, and the alphabet comes out the wrong size. After
that nothing downstream can recover -- the traces are written in a language the
stored programs do not speak -- and it is silent, because a wrong alphabet
produces low accuracy rather than an error.

The first sweep showed exactly this: at noise 0.05 and above, four- and
six-symbol rooms reported **eight** letters and were unusable. The eight-symbol
room reported eight too, and *looked* fine while its clusters were wrong,
because the cluster cap happened to equal the truth. That near-miss is why the
cap is now far above any alphabet being tried: a saturated count must not be
able to read as success.

**This is the third fixed constant in this session that had to become relative
to the noise**, after the regime tracker's reuse allowance and its detection
reference. The shape is always the same: measure the noise instead of assuming
it. Pairwise distances are bimodal -- same stimulus, different stimulus -- so
the boundary is the largest gap between the modes and the tolerance is its
midpoint. With that, all twelve readable cells recover the alphabet exactly.

## Refusing, rather than guessing

At noise 0.15 and above the estimator returns nothing and the agent stops. That
is the correct answer rather than a limitation: at 0.20 the largest
within-stimulus distance is 5.39 and the smallest between-stimulus distance is
5.55. The stimuli genuinely overlap, and an agent that named an alphabet anyway
would be guessing and would then be confidently wrong about everything.

The margin that decides this is itself swept rather than chosen, over 63 cases
of three alphabet sizes, seven noise levels and three observation seeds, against
the failure that actually matters:

| margin | alphabet exact | **wrong but accepted** | refused |
| ---: | ---: | ---: | ---: |
| 0.30 | 37 | **0** | 26 |
| **0.25** | **38** | **0** | **25** |
| 0.20 | 38 | **0** | 25 |
| 0.15 | 42 | 2 | 19 |
| 0.10 | 44 | 5 | 14 |
| 0.05 | 45 | 7 | 11 |

Loosening to 0.15 buys four more readable rooms and two silent corruptions.
0.25 and 0.20 are indistinguishable here, so the more conservative one stays.

## What is honestly weak

**The refusal boundary is the frontend's, not the agent's.** Everything above
0.10 fails at a clustering step that happens before any learning. A frontend
trained to be invariant to this noise would move the boundary, and nothing here
attempts that -- the frontend is frozen and curated.

**Noise was added to pixels, not to the task.** The symbol sequence and the
rule are untouched, so this tests perception rather than the world being
genuinely less predictable. Label noise was measured separately and survives to
20%; the two have not been combined.

**Still one modality and one frontend.** The audio stream exists in the
configuration and is not exercised. Nothing here shows the alphabet estimator
works on a different encoder, and its threshold was swept on this one.

**Twelve tasks per cell, one seed.** The direction is consistent across all
twelve cells, but this is a development-seed diagnostic and not a holdout
measurement.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.environment_widening
```

About eighty seconds.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_environment_widening.py -q
```
