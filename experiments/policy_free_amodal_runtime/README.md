# Policy-free amodal runtime

This pressure test verifies the new canonical execution seam inspired by the
exported games session:

```text
opaque events -> one frozen amodal controller -> opaque state
             -> factual model search toward an opaque goal
             -> intention bus -> independent decoder
```

The controller's direct intention is measured as a control but is never sent
to the decoder. The factual model consumes one transition bundle through
replay-free sufficient statistics, and the planner derives behavior for four
novel goals. Model immutability during search, exact persistence, controller
freezing, goal shuffling, and a random floor are explicit gates.

This promotes runtime wiring and policy-free behavior derivation only. It does
not claim learned state grounding, unrestricted planning, arbitrary new
computation, or general continual learning.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/policy_free_amodal_runtime/train.py \
  --seed 85001 \
  --report-out /tmp/policy-free-amodal-85001.json
```
