# Policy-free intention repertoire

This pressure test removes the caller-authored candidate-intention tensor from
the policy-free runtime. An external append-only repertoire is populated by a
short opaque experience stream; execution then retrieves those vectors and
adds only the current controller seed as an ephemeral exploration candidate.
The factual transition model still derives behavior by goal-conditioned search.

This is a candidate-discovery and persistence milestone, not a claim of
general continual learning. The repertoire does not become a reward-ranked
policy: verifier outcomes are retained as sufficient statistics, while the
planner remains responsible for selecting behavior from factual predictions.

Run one seed with:

```bash
PYTHONPATH=. .venv/bin/python experiments/policy_free_intention_repertoire/train.py \
  --seed 85101 \
  --report-out /tmp/policy-free-intention-repertoire-85101.json
```
