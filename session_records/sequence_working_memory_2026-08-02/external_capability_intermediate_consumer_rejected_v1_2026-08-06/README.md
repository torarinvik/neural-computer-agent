# Head-only external consumer — 2026-08-06

Status: rejected as a positive-transfer promotion; retained as a successful
intermediate-only capability diagnostic.

The head program first learned `complement4` and was frozen. A downstream
consumer then learned the novel `complement_reverse4` target while the
pipeline hid raw events from every program after the head. The consumer saw
only the prior opaque intention, opaque feedback, and scalar outcomes. A
matched fresh two-program pipeline was trained under the same head-only
contract.

| metric | seed 69316 |
| --- | ---: |
| head stable bits | 6,144 |
| consumer stable bits | 16,384 |
| fresh stable bits | 6,144 |
| consumer accuracy | 0.8633 |
| fresh accuracy | 1.0000 |
| blank accuracy | 0.5586 |
| reward-shuffled accuracy | 0.4453 |
| zero-head accuracy | 0.5859 |
| zero-consumer accuracy | 0.5391 |

This is important positive architectural evidence: a consumer can learn and
execute a new computation without downstream raw-event access, and both the
head and consumer are causal. It is not yet a continual-learning gain: the
fresh learner reaches mastery in fewer verifier bits, so inherited head-only
state is currently a sample-efficiency liability. The next high-ROI direction
is curriculum or prior initialization that preserves the opaque intermediate
contract while making the consumer learn faster.

Reload, corruption rejection, frozen-core, shuffled-outcome, retention, and
zero-replay controls pass. Full accounting is in
`sample_efficiency_ledger.json` and `report_seed69316.json`.
