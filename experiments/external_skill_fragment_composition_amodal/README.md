# External skill fragment composition

This is the replicated pressure test for the compositional fragment-bank
boundary. A parent amodal controller is trained on forward reproduction and
then frozen. A shared external register interpreter and one coefficient row
acquire `reverse` from fresh rendered events; a second coefficient row then
acquires `rotate` without replaying the first task. The held-out
`reverse -> rotate` procedure is learned by an external trace combiner and
decoder over the frozen serial chain. A matched fresh interpreter learns the
same held-out procedure from scratch.

The bank stores short coefficient sequences over one shared instruction basis.
It does not store task-sized neural modules, and the controller never receives
fragment indices, operation names, correct actions, or verifier targets. The
trainer uses operation names only to select the local rendered verifier and to
construct controls. The deployed path receives learned event tensors,
feedback, and opaque intention payloads.

The audit is diagnostic until it passes all gates on replicated seeds:

- old `reverse` retention after the new fragment is acquired;
- held-out serial composition beats a matched fresh learner in stable
  verifier-bit cost, measured at repeated held-out prefixes;
- reversed fragment order and shuffled outcomes fail the composition gate;
- zeroed fragment codes fail, proving the decoder is not using register content
  as a fragment bypass;
- frozen parent/controller digest is unchanged;
- no replayed examples;
- fragment routing remains row-permutation equivariant;
- exact bank persistence/reload and checksum rejection.

The 128-update audit passed on seeds 69316 and 69317. Inherited composition
reached stable mastery in 6,144 and 9,216 verifier bits; matched fresh learners
needed 24,576 and 12,288 bits. The replicated geometric-mean advantage is
therefore positive, but this establishes only bounded reusable composition—not
arbitrary program induction, unrestricted memory growth, or general continual
learning.

The important implementation lesson is that the fragment code is not itself a
complete intention. The interpreter emits an ordered execution trace, and a
small external combiner learns how to read that trace before the output decoder
acts. This keeps the controller frozen while preserving a trainable path for
composition-specific credit assignment. Code materialization is normalized at
the external boundary so coefficient/basis scale does not silently turn every
fragment into a near-zero instruction.

Run a short smoke rung with:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/external_skill_fragment_composition_amodal/train.py \
  --parent-updates 8 --primitive-updates 8 --composition-updates 8 \
  --batch-size 32 --audit-count 128 --eval-every 4 \
  --report-out /tmp/fragment-composition-smoke.json
```

Reproduce the promoted audit with two matched seeds:

```bash
for seed in 69316 69317; do
  PYTHONPATH=src:. .venv/bin/python -m \
    experiments.external_skill_fragment_composition_amodal.train \
    --seed "$seed" --parent-updates 64 --primitive-updates 64 \
    --composition-updates 128 --batch-size 32 --span 3 --audit-count 128 \
    --eval-every 16 --report-out "/tmp/fragment-composition-$seed.json"
done
```

## Four-fragment closure audit

`train_multi.py` extends the same boundary to four sequentially acquired
fragments: `reverse`, `rotate`, `complement`, and `prefix_parity`. Primitive
acquisition is deliberately isolated from composition acquisition. A fragment
must first earn stable held-out mastery on its own fresh verifier outcomes;
only then is it protected and used by the held-out composition stage. This
separation is important: sharing one decoder objective between a primitive and
its longer composition entangled the stored primitive and caused a replicated
seed failure.

The exact 256-update acquisition audit passed on seeds 69316 and 69317. Each
seed retained all four primitives, mastered the held-out order
`prefix_parity -> complement -> reverse -> rotate`, rejected reversed order,
zero codes, missing sequence evidence, and shuffled outcomes, and preserved the
frozen parent. Inherited composition reached stable mastery in 6,144 bits on
both seeds; matched fresh learners required 12,288 bits on both seeds.

```bash
for seed in 69316 69317; do
  PYTHONPATH=src:. .venv/bin/python -m \
    experiments.external_skill_fragment_composition_amodal.train_multi \
    --seed "$seed" --parent-updates 64 --primitive-updates 256 \
    --composition-updates 64 --batch-size 32 --span 3 --audit-count 128 \
    --eval-every 32 --report-out "/tmp/fragment-multi-$seed.json"
done
```

## Multi-target frozen-bank closure audit

The next rung reuses the same four acquired fragments across three independently
held-out orders. Each order receives a fresh external trace combiner and output
decoder, while the acquired register machine and fragment bank remain frozen.
This tests whether the gain belongs to reusable external capability rather than
one target-specific decoder or continued memory growth.

The matched 128-update audit promoted on seeds 69316 and 69317. Every target
reached stable mastery through the inherited frozen-bank path at 6,144 verifier
bits; matched fresh learners required 12,288 bits for every target. The bank
checksum was identical before and after target learning. Wrong-order,
zero-fragment, missing-evidence, and reward-shuffled controls were all rejected,
and no examples were replayed.

The intermediate 64-update replication is retained as a rejected diagnostic:
seed 69317's third fresh control ended at 0.75 and had no stable prefix. Doubling
only the composition exposure resolved that variance without changing the
architecture or relaxing any gate.

```bash
for seed in 69316 69317; do
  PYTHONPATH=src:. .venv/bin/python -m \
    experiments.external_skill_fragment_composition_amodal.train_multi \
    --seed "$seed" --parent-updates 64 --primitive-updates 256 \
    --composition-updates 128 --batch-size 32 --span 3 --audit-count 128 \
    --eval-every 32 --report-out "/tmp/fragment-multi-target-$seed.json"
done
```

This is still bounded continual-memory/composition transfer. It does not
establish arbitrary program induction, unrestricted memory growth, compression,
or general continual learning.

## Shared-composition pressure test

`train_shared_multi_target.py` deliberately removes the fresh combiner per
target. One segment-aware external learner and one decoder train across three
orders while three other orders remain held out. The rich trace carries learned
codes, transition deltas, and segment lengths; `ExternalSkillFragmentLearnerTrace`
removes route indices and route scores before the learner sees it.

The 128-update seed-69316 rung was rejected: training accuracy was
`0.6536/0.9531/0.7760`, held-out accuracy was `0.6276/0.5182/0.6094`, and no
stable prefix was reached. Controls and frozen-bank/persistence gates passed.
The batched variable-length transport path reduced wall time from 496.5 to
353.5 seconds, but that is an implementation gain, not a capability gain. The
decision record and accounting ledger are in
`session_records/external_skill_fragment_shared_multi_target_v2_2026-08-11/`.

`--combiner-mode operator` swaps in one code-conditioned low-rank state
transition shared across every opaque segment and composition depth. The matched
seed-69316 run was rejected: training accuracy was
`0.6849/0.7266/0.7786`, held-out accuracy was `0.6016/0.5833/0.7083`, and
wrong-order accuracy remained `0.6563/0.6745/0.7214`. No stable prefix was
reached. Frozen-parent, frozen-bank, zero-code, missing-evidence,
reward-shuffled, persistence, and zero-replay controls passed, but the operator
did not improve verified capability. The full decision record is in
`session_records/external_skill_fragment_operator_algebra_rejected_2026-08-11/`.

The operator ABI and atomic checksummed persistence are retained as
infrastructure, not as a promoted learned composition law. The next experiment
should isolate ordered credit assignment with a smaller contrastive curriculum
before scaling depth or adding more memory capacity.

The follow-up order-contrast diagnostic is also rejected. It first corrected
the variable-depth transport by grouping programs with equal executable
lengths and making target rows contiguous, then reused each example with a
cyclically shifted route and a trainer-only inverted counterfactual loss. At a
matched seed-69316 16/64/64 audit, the baseline reached mean held-out accuracy
`0.5972`; the contrast arm reached `0.5208` while consuming 576 additional
paired rollouts. It improved wrong-order rejection but reached no stable
prefix, so it is retained only as an optional diagnostic hook. The corrected
reports and ledger are in
`session_records/external_skill_fragment_order_contrast_rejected_2026-08-11/`.

```bash
PYTHONPATH=src:. .venv/bin/python -m \
  experiments.external_skill_fragment_composition_amodal.train_shared_multi_target \
  --seed 69316 --parent-updates 64 --primitive-updates 256 \
  --composition-updates 128 --batch-size 32 --span 3 --audit-count 128 \
  --eval-every 32 --report-out /tmp/fragment-shared-multi-target.json
```

## Append-only depth-growth pressure test

`train_depth_growth.py` tests the CPU-plus-files boundary directly. One
jointly trained atomic foundation is frozen; a single shared
`ExternalSkillFragmentGrowthCombiner` appends one zero-impact,
trace-conditioned slot per composition depth; and only the new slot learns
from fresh verifier outcomes at each rung. The parent controller, register
interpreter, and acquired fragment bank are frozen on the inherited path.
Earlier slots are protected after each rung, so this is a real no-replay
growth transaction rather than a fresh target-specific adapter.

The initial seed-69316 depth-2 diagnostic reached a minimum accuracy of
`0.9167` across all 12 ordered pairs after 256 slot updates. It is not yet a
promotion: deeper depths, held-out program transfer, repeated retention, and
replication remain required. The report also distinguishes training exposure
from audit exposure.

```bash
PYTHONPATH=src:. .venv/bin/python -m \
  experiments.external_skill_fragment_composition_amodal.train_depth_growth \
  --joint-foundation --seed 69316 --parent-updates 64 \
  --foundation-updates 128 --stage-updates 256 --batch-size 32 --span 3 \
  --audit-count 128 --eval-every 32 \
  --report-out /tmp/fragment-depth-growth.json
```

The sequential shared-decoder acquisition control remains rejected because
separately learned atomic representations were not aligned for a common
readout. The correct next architectural pressure is therefore deeper
append-only growth after a shared foundation, not more target-specific
decoders.
