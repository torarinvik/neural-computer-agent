# Three-way ranked memory-read race — pre-registration

## Why this primitive

The controller already chooses between the primary and second-ranked physical
memory reads. The next gradual primitive adds a third operation: use the
third-ranked candidate. No task name, correct action, unattempted outcome, or
semantic label is exposed.

A zero-training viability probe at capacity 6 found:

- third-ranked read is oracle-optimal in `12.16%` of contexts;
- current champion utility is `0.6489`;
- oracle utility is `0.7946`;
- champion-to-oracle headroom is `14.57` points.

This is a substantially better learning surface than capacity/cost shifts,
whose headroom was only 1.8–2.45 points.

## Horse race

Two matched pairs distinguish archive reuse from global-champion reuse:

1. **Stored transferred/reset (width 64):** preserve the stored two-action latent trunk and its two
   learned action values; initialize the unseen third value to their symmetric
   mean, versus the same architecture reset.
2. **Champion transferred/reset (width 16):** preserve the actual global
   champion's binary advantage exactly as `Q1-Q0`, add a neutral third value,
   versus the same architecture reset.

Each observes only four generic memory statistics, a uniformly randomized
attempted operation, and its scalar outcome. Both get equal verifier-bit and
optimizer-update budgets.

## Sub-minute pass gate

- Absolute target: immutable champion utility + 2 points on a disjoint private
  stream.
- Transferred arm must cross the target.
- It must cross in fewer verifier bits than reset.
- It must finish above the immutable champion.
- Oracle must remain at least 2 points beyond the target.

Private evaluation may measure the race but cannot affect gradients, selection,
or promotion. Passing this rung permits a fresh-seed replication and then an
independent 2,400-outcome confirmation; it is not itself a durable promotion.

## Exploratory result and replication gate

The 960-bit seed-8043 run rejected the weaker stored child as an immediate
zero-shot policy, but showed the direct champion expansion approaching the
target. Extending on fresh seed 8044 reversed the architectural ranking:

- direct champion expansion became unstable and was rejected;
- stored action-value transfer crossed the target at 1,080 bits;
- matched width-64 reset crossed at 1,320 bits;
- transferred final utility was `0.70077` versus champion `0.65860`;
- transferred paired-IPS lower 95% bound was `+0.0712`.

This is an exploratory 18% samples-to-target advantage, not yet a breakthrough.
One fresh-seed replication is pre-registered. It passes only if:

1. stored transfer crosses the champion+2-point target;
2. it crosses at least one 60-bit update before matched reset;
3. final transferred utility remains at least 2 points above champion; and
4. its final paired-IPS lower 95% bound against champion is positive.

If it passes, the candidate advances to frozen independent confirmation and
adversarial feature/action audits. If any condition fails, do not scale.

Seed 8045 failed condition 2: transfer crossed at 3,120 bits while reset crossed
at 600, despite transfer finishing above champion with a positive lower bound.
The unrestricted transferred models repeatedly destroy inherited behavior early
and relearn it later.

One final sub-minute mechanism test is allowed: freeze the global champion's
binary encoder and decision output, add a separately trainable third-action
value, and spend at most 960 verifier bits. This arm passes only if it preserves
champion utility at initialization, exceeds champion+2 points within the budget,
and beats the matched width-16 reset's samples-to-target. A failure closes the
structured-reuse branch.

Seed 8046 exposed a value-scale interface error: the frozen champion provided a
relative advantage around a fixed `0.5`, while the new output learned an
absolute reward value. Its numerically incomparable third value dominated
73.8% of contexts and collapsed utility. One correction is permitted because
`argmax` over incomparable scales is invalid: learn a small shared baseline and
positive advantage scale from attempted outcomes, while the frozen champion
still determines the ordering of actions 0/1. The 960-bit gate and fresh seed
remain unchanged.

Seed 8047 calibrated the scales but did not learn useful third selection within
960 bits. The representation audit found the curriculum flaw: the original four
statistics describe candidate one and its margin to candidate two, but contain
no candidate-three-specific evidence.

The next representation adds only generic physical evidence: second-to-third
score margin and usage strengths for candidates two and three. The stored
two-action Q head remains frozen on the original first four values. A
zero-initialized residual sees all seven and learns only the third Q value.
Matched reset sees the same seven values and has the same width-64 budget. This
is a new, pre-registered compositional-reuse test rather than an extension of a
failed run.

Seed 8048 improved over reset but missed the +2-point gate. The final tiny
composition test combines two verified assets without changing either: the
stored action-value head supplies calibrated center/magnitude; the stronger
champion supplies the ordering of old actions 0/1; a new residual learns action
2 from seven generic statistics. It gets the same 960-bit gate once.

Seed 8049 failed. The flat three-action formulation is closed.

## Gradual option-composition fork

The task is now factored into the smallest new decision. The verified champion
is one opaque latent option that internally selects old physical action 0/1;
the only alternative is physical action 2. A new two-value head learns
`old-option` versus `third-read` from randomized attempted options and scalar
outcomes. A flat width-32 three-action reset receives the same verifier-bit and
update budget.

At initialization the composed arm is bit-exact to the champion. The 960-bit
gate is unchanged: champion+2 points, earlier than flat reset, and sustained at
the final checkpoint. This tests whether reusing a learned option reduces the
new search space.

Seed 8050 gained 1.01 points but missed the +2 target; flat reset crossed at 960
bits. A discarded-weight dense probe then established that the seven statistics
are sufficient: 97.4% held-out decision accuracy and `0.7875` utility versus the
champion's `0.6625`.

The blocker is sparse-outcome utilization. Both arms now receive four replay
optimizer minibatches per newly verified batch, sampled only from their own
previously attempted outcomes. Verifier bits remain 960 per arm; compute
accounting records all 64 optimizer updates. The pass gate is unchanged.

Seed 8052 passed strongly:

- option composition crossed champion+2 at 120 verifier bits;
- flat reset first crossed at 360 bits;
- option final utility was `0.72772`;
- champion utility was `0.65276`;
- flat reset finished at `0.65685`;
- each arm used 960 verifier bits, 64 replay updates, and about 0.31 seconds.

Fresh-seed replication is pre-registered with the identical configuration. It
must cross earlier than flat reset, finish at least 2 points above its champion,
and remain above target at the final checkpoint. Passing advances to frozen
independent outcome confirmation plus feature-shuffle, option-disable, and
retention audits.

## Breakthrough result

The replication and complete audit passed.

### Sample-efficiency races

| Seed | Option target | Flat-reset target | Option final gain |
|---|---:|---:|---:|
| 8052 | 120 bits | 360 bits | `+7.50` points |
| 8053 | 840 bits | 960 bits | `+5.28` points |

Both arms received equal verifier-bit budgets. Replay used only previously
observed attempted outcomes and cost 64 optimizer updates per arm.

### Frozen independent audit

The seed-8053 option was frozen before audit. On 2,400 new uniformly randomized
attempted options:

- estimated improvement: `+5.35` points;
- standard error: `0.48` points;
- lower 95% confidence bound: `+4.40` points.

Across eight additional fresh streams:

| Policy/control | Mean utility |
|---|---:|
| immutable champion | `0.66230` |
| composed option | `0.71491` |
| shuffled option features | `0.57900` |
| reversed option decision | `0.54375` |

The composed option improved all eight streams. Feature shuffling cost 13.59
points and reversing its decision cost 17.12 points, ruling out an inert option
or simple action-frequency artifact. Binary mapping and four-rule retention
both passed.

### Durable skill

The verified option was committed atomically as child
`16085614999f54495580` of champion `b42abe0d6d427df30be6`:

- 960 training verifier bits;
- 2,400 fresh confirmation bits;
- bit-exact reload;
- SHA-256 verification;
- deliberate child corruption detected;
- parent remained retrievable after child corruption.

This is the first verified compounding-learning result in the project: a prior
latent skill was retrieved as an opaque option, reduced a new three-action
problem to one new binary decision, learned the extension in fewer verifier
bits than a flat reset on two independent streams, beat the global champion,
survived causal audits, retained old skills, and persisted with honest lineage.

The mechanism is general and task-agnostic: preserve a verified primitive,
learn a small router over that primitive plus a new operation, and replay scarce
verified outcomes. No task name, correct operation, unattempted outcome, or
semantic label entered training.
