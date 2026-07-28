# Why transfer stops, and whether it can be restarted

Transfer in this ladder is a first-composition effect: ancestry 1 → 2 makes the
next skill much cheaper, 2 → 3 and 3 → 4 do nothing. This record is about the
mechanism behind that, which turns out to be structural rather than statistical.

## The measurement

A new slot's input is `cat([state.hidden, event])`. Comparing what that input
actually contains across the ancestry, on the same lifetimes:

| parent | distance from the 3-skill parent's features |
|---|---:|
| 1 skill | 2.472511 |
| 2 skills | 0.014051 |
| 3 skills | 0.000000 |
| 4 skills | **0.000000** |
| 5 skills | **0.000000** |

The shared base is bit-identical across every level (44 of 44 tensors), so the
only way ancestry can reach a new slot's input is through behavior: an earlier
slot changes the logits, which changes the action, which feeds the recurrent
state. A rectified gate that is exactly shut on a foreign task's events changes
none of those. So a three, four and five skill parent hand the next slot the
same numbers, to the last bit.

**A new slot cannot inherit anything, because nothing about the ancestry reaches
its input.** That is not a weak effect being missed by an underpowered test; it
is zero by construction.

## What this unifies

The two previous findings were recorded as separate results. They are the same
result seen from two sides:

| gate | rung-4 transfer | neighbour retention |
|---|---:|---|
| sigmoid, never exactly zero | +0.0116 | degrades, −0.0059 to −0.0132 |
| rectified, exactly zero | −0.0163 | exact, deltas at 0.0000 |

Transfer and interference were never separate phenomena. Both travel the one
channel — an earlier slot influencing behavior on the new task's events. Opening
it buys a little transfer and costs retention; closing it makes retention exact
and makes transfer impossible. The earlier framing, that removing interference
would let compounding show itself, had the causality backwards: removing
interference is what guaranteed there would be none.

## The fix under test

Separate reading from writing. An earlier slot's gate should decide whether it
*speaks*, not whether it can be *consulted*. `skill_adapter_reads_prior` gives a
new slot the earlier slots' pre-gate hidden layers as extra input while leaving
their writes gated exactly as before.

Two properties were checked before spending compute on it:

- inserting a reading slot is still exactly behavior-preserving — the rollout
  logits are unchanged, so the retention guarantee is untouched;
- a trained earlier slot now changes the next slot's input, where previously it
  could not, even when that earlier slot's gate was fully shut.

The experiment is ancestry 3 → 4 with and without the read path, twelve seeds
over six budgets, otherwise identical. Ancestry 3 → 4 was chosen because it
previously measured −0.0002 at p = 0.44 — as close to exactly nothing as the
design produces — so any real recovery is unambiguous.

## Reading the result

The reference points, same regime:

- ancestry 1 → 2 transfers at +0.065 to +0.087, p < 1e-7, in both task families;
- ancestry 3 → 4 without a read path: −0.0002, p = 0.44.

A read path that restores compounding should move the 3 → 4 cell toward the
1 → 2 range while the retention deltas stay at zero. A read path that does not
would say the information a later slot needs is not in an earlier slot's hidden
layer, and the next thing to try is what a slot is allowed to write to, not what
it is allowed to read.
