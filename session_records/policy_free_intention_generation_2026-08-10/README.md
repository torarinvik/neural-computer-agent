# Policy-free intention generation — 2026-08-10

This record promotes the first replicated end-to-end rung for the new
memory-side intention-generation seam. A frozen amodal controller produces an
opaque state context; an external stochastic generator proposes a continuous
opaque intention; a scalar factual verifier supplies outcome-only credit; and
verified content is admitted to the stable external intention repertoire.
`PolicyFreeAmodalRuntime` then plans with the admitted candidates.

The experiment also exercises the files-like growth contract. A source cell is
trained, protected, and copied into a successor cell before successor-only
outcomes train the new cell. The source remains byte-stable. A matched fresh
learner and a reward-shuffled control separate transfer from mere capacity or
random drift.

Both seeds pass every gate:

| seed | source bits | successor bits | fresh bits | successor/fresh updates | shuffled score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 85101 | 19 | 9 | 13 | 9 / 13 | 0.0044 |
| 85102 | 32 | 11 | 20 | 11 / 20 | ~0 |

The successor is therefore faster than a fresh learner in both seeds, with
fresh-over-successor transfer ratios of `1.44x` and `1.82x`. The controller and
state adapter remain frozen, replay is zero, exact generator/repertoire reload
passes, and both generated intentions become planner-usable only after held-out
admission.

The claim boundary is deliberately narrow: this is a replicated bounded
continuous-intention discovery and external-memory growth result on a synthetic
factual verifier. It is not general continual learning, unrestricted memory
growth, arbitrary program induction, or evidence that the controller can learn
new computation from memory alone. The next pressure test should use partial
multimodal contexts, multiple competing memories, delayed/noisy outcomes,
reversals, and a longer sequence of append/protect cycles.

Reproduce with:

```bash
.venv/bin/python experiments/policy_free_intention_generation/train.py \
  --seed 85101 \
  --report-out /tmp/policy-free-intention-generation-85101.json
```

Reports:

- `report_seed_85101.json`
- `report_seed_85102.json`
- `sample_efficiency_ledger.json`
- `SHA256SUMS`
