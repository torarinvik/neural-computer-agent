# Variable-prefix online context identity

This rung uses the promoted disjoint-dynamics fixture but trains the external
context encoder with noisy prefixes of source evidence. The online router must
admit each novel regime after only seven of fourteen transition rows, then
continue adapting the selected external model as later rows arrive.

The controller remains frozen and behavior is still derived by opaque
model-based search. Existing source slots are read-only; only a currently
selected novel slot receives online updates. The report separates prefix
admission evidence, current-stream updates, old-slot updates, planner mastery,
and matched fresh-model work.

This is a streaming-boundary pressure test, not a general continual-learning
claim. It uses a finite transition-table fixture and a fixed prefix window.

```text
.venv/bin/python experiments/external_partial_evidence_identity/train.py \
  --seed 70511 \
  --report-out /tmp/external-partial-evidence-identity.json
```
