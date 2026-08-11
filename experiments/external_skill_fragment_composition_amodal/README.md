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
