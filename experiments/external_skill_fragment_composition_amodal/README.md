# External skill fragment composition

This is the first empirical pressure test for the compositional fragment-bank
boundary. A parent amodal controller is trained on forward reproduction and
then frozen. A shared external register interpreter and one coefficient row
acquire `reverse` from fresh rendered events; a second coefficient row then
acquires `rotate` without replaying the first task. The held-out
`reverse -> rotate` procedure is learned by a fresh decoder over the frozen
serial chain. A matched fresh interpreter learns the same held-out procedure
from scratch.

The bank stores short coefficient sequences over one shared instruction basis.
It does not store task-sized neural modules, and the controller never receives
fragment indices, operation names, correct actions, or verifier targets. The
trainer uses operation names only to select the local rendered verifier and to
construct controls. The deployed path receives learned event tensors,
feedback, and opaque intention payloads.

The audit is diagnostic until it passes all gates on replicated seeds:

- old `reverse` retention after the new fragment is acquired;
- held-out serial composition beats a matched fresh learner in stable
  verifier-bit cost;
- reversed fragment order and shuffled outcomes fail the composition gate;
- zeroed fragment codes fail, proving the decoder is not using register content
  as a fragment bypass;
- frozen parent/controller digest is unchanged;
- no replayed examples;
- fragment routing remains row-permutation equivariant;
- exact bank persistence/reload and checksum rejection.

Passing this audit would establish bounded reusable composition, not arbitrary
program induction, unrestricted memory growth, or general continual learning.

Run a short smoke rung with:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/external_skill_fragment_composition_amodal/train.py \
  --parent-updates 8 --primitive-updates 8 --composition-updates 8 \
  --batch-size 32 --audit-count 128 --eval-every 4 \
  --report-out /tmp/fragment-composition-smoke.json
```
