# Online disjoint-dynamics model routing

This pressure test removes the nested `position + delta` structure from the
policy-free model-bank line. Every regime has the same opaque state and
intention widths, but a different transition table. The router sees only
single transition rows, buffers them outside the bank, and uses the learned
context encoder plus factual prediction error to admit or reuse an opaque
model slot.

After admission, behavior is derived by `ExternalModelBasedPlanner` against
held-out opaque goals. The controller is frozen. Existing slots are never
updated during later regimes, so retention is checked both behaviorally and
by parameter digest. Fresh-model controls, wrong-context factual error,
route-permutation diagnostics, persistence, and no-old-replay accounting are
included.

This is a fast adversarial rung, not a general continual-learning claim. The
transition tables are finite and supplied by the verifier; the context
encoder is pretrained on two source regimes; and the planner still uses a
finite horizon. Promotion means the factual-model route survives genuinely
disjoint dynamics under the current interface.

```text
.venv/bin/python experiments/external_disjoint_dynamics_online/train.py \
  --seed 70411 \
  --report-out /tmp/external-disjoint-dynamics-online.json
```
