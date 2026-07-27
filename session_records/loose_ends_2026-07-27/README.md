# Closing the rectified-gate loose ends

Three things were left open when exact-zero gating removed per-rung
interference: the dead-gate failure rate, the unreached one-support form of the
composition rung, and whether the result survives at depth. Two are closed, one
is a clean negative, and the depth work produced a sharper law than the one it
set out to test.

## 1. Dead gates: solved by holding the gate open, not by leaking

A rectified gate that shuts everywhere before its adapter has learned anything
is left without gradient and stays shut. The obvious fix — a small leak below
the knee, annealed to exactly zero so the final gate is still exactly closable —
**does not work**:

| initial leak | seeds | dead | new skill (live) | worst delta (live) |
|---|---:|---:|---:|---:|
| 0.0 | 24 | 1 (4.2%) | 0.9932 | −0.00342 |
| 0.02 | 24 | 1 (4.2%) | 0.9772 | −0.00168 |
| 0.05 | 24 | 1 (4.2%) | 0.9806 | −0.00163 |

The death rate does not move, and the reason is visible in the data: **the same
seed (8507) dies at all three leak settings.** Death is deterministic in the
initialisation, not a stochastic event a leak can rescue. The leak does halve
the residual degradation, at a small cost in new-skill accuracy, so it is worth
keeping for that reason alone — but it is not the fix.

Holding the gate **open and frozen** for the first few percent of the budget is.
A gate cannot shut on an adapter that still outputs zero if it cannot move at
all:

| gate warmup | seeds | dead | seed 8507 | new skill (live) | gates |
|---|---:|---:|---|---:|---|
| 0 updates | 4 | 1 | **dead** | 0.9769 | 3/3 |
| 154 (5%) | 4 | **0** | alive | 0.9899 | 4/4 |
| 307 (10%) | 4 | **0** | alive | 0.9912 | 4/4 |
| 768 (25%) | 4 | **0** | alive | 0.9916 | 4/4 |

It rescues the deterministic failure and *improves* new-skill accuracy. Five
percent is enough. The earlier "about 8%" death rate was one failure in twelve;
across 24 seeds the true rate was 4.2%, and with warmup it is zero in every cell
measured.

The report now also carries an explicit `slot_dead` flag and refuses to promote
a dead slot, because a slot shut on 100% of its own task's events reads as
perfect retention while having learned nothing.

## 2. The one-support composition rung: not reached, with a reason

`contextual_composition` is mastered at two support outcomes and remains
unreached at one. Graduated reduction — start at two, drop to one partway, which
is how `binary_mapping` was originally acquired — does not close it:

| arm | steps | new skill at one support | gates |
|---|---:|---:|---|
| straight to one support | 6144 | 0.5809 | 0/3 |
| graduate at 25% | 12288 | 0.6179 | 0/2 |
| graduate at 50% | 12288 | 0.5958 | 0/2 |
| graduate at 75% | 12288 | 0.5375 | 0/1 |

Every arm is far below the 0.85 gate, and none of the schedules beats going
straight there. One support is information-theoretically sufficient — a single
outcome identifies the hidden rule given the visible identity and context — so
this is a credit-assignment failure, not an identifiability one.

There is also a structural reason the obvious workaround cannot work. Adding the
one-support form as its own later rung is impossible: **the support count is not
visible in the frame.** A one-support and a two-support composition lifetime
render identically, so a slot gate — which reads only the frozen encoder's view
of the frame — cannot be selective between them. Any slot that learns the
one-support form must perturb the two-support skill. The one-support form has to
be reached inside the rung that acquires the skill, and at these budgets it is
not.

## 3. Depth: interference is set by cue separability, not by depth

The chain reached depth 5 with every rung promoted on its own gates.

| rung | added | new skill | inherited deltas | worst | exactly zero |
|---|---|---:|---|---:|---|
| 4 | `contextual_composition` | 0.9854 | −0.0012, +0.0016, +0.0005 | −0.0012 | 0/3 |
| 5 | `context_rule_xor` | 0.9944 | 0.0000, 0.0000, 0.0000, **−0.0283** | −0.0283 | **3/4** |

Rung 5 leaves three of its four inherited skills at **exactly** zero change and
damages only one — `contextual_composition`, the single most similar earlier
skill. That is not depth-driven decay; it is a specific pairwise effect, and the
mechanism turned out to be measurable.

The slot's statistics at rung 5, before the cue was fixed:

| skill | exactly shut | residual norm | delta |
|---|---:|---:|---:|
| `visible_context_xor` | 84.1% | 0.21 | +0.0007 |
| `binary_mapping` | 65.4% | 0.53 | −0.0002 |
| `visible_context` | 59.8% | 0.59 | −0.0024 |
| `contextual_composition` | 37.6% | 2.49 | −0.0303 |

Degradation tracks the residual the slot leaves, and the slot cannot shut on a
skill the **frozen encoder cannot distinguish**. Measuring that separation
directly explained everything:

| cue pair | frozen-feature separation | measured interference |
|---|---:|---:|
| `visible_context_xor` vs `composition` | 4.05 | +0.0007 |
| `override` vs `composition` | 1.36 | ~0 |
| `composition` vs `context_rule_xor` | **0.25** | **−0.0303** |

The cause is `VisionEventEncoder`'s closing `AdaptiveAvgPool2d(1)`. A global
average pool discards position, and those two cues were both fifteen-pixel bars
on the bottom row differing only in horizontal offset — nearly identical to that
encoder. Widening one cue so the pair differs in **area** rather than position
raised the separation from 0.25 to 1.18, and `context_rule_xor` then learned and
promoted where it had previously stalled at 0.70.

**A slot can only be as selective as the frozen features allow.** Cue slots must
be separable under the encoder that will actually read them, which for a global
average pool means separating by area, not by position.

`probe_cue_separability.py` makes this a preflight check: it encodes the same
events under each operation's cue against a candidate parent and reports the
pairwise separations, so a rung that will interfere is identifiable before any
training compute is spent. On the current four-skill controller one pair still
sits at the warning threshold — `visible_context` against
`contextual_composition` at 0.9998 — which is exactly the pair that has shown
small but nonzero degradation throughout.

## State and how to resume

Promoted and pulled back: `artifacts/checkpoints/depth/depth_rung4_8600.pt` and
`depth_rung5_8600.pt`, both passing every gate.

Left running on the rented box when this session ended, results not yet
collected:

- rung 6 (`contextual_override`) of the depth chain, via `/workspace/depth.sh`,
  which is resumable and skips rungs whose checkpoint already exists;
- a 24-seed panel at 5% gate warmup, to confirm the zero death rate at the same
  scale the 4.2% figure was measured at;
- a 24,576-update graduated one-support attempt, the last budget worth trying
  before that rung is called closed.

That box has no host volume, so nothing there survives a recycle. Everything
above is in this record.
