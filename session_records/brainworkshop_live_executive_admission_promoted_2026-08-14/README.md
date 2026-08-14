# Live verifier-gated executive admission

This session closes the next growth bottleneck: a proposed frozen executive can
be evaluated through the same live `RECEIVE -> think -> EMIT` boundary as an
admitted skill, then enter `AgentBrain.bank` only after a stable verifier gate.

The promotion lane evaluated a delay-2 executive on three independent n-back
lifetimes (24 unique verifier bits, three logical lifetimes). Each lifetime was
perfect. The first two observations left the bank unchanged; the third admitted
the artifact into slot 0. The controller, decoder, and executive program were
all frozen, with zero optimizer updates and zero replayed examples.

The negative lane evaluated a delay-1 executive against private delay-2
verification (24 bits across three lifetimes). Its outcomes were `0.50`,
`0.375`, and `0.50`; the candidate was rejected and the bank digest remained
unchanged. The promoted bank was then saved and reloaded exactly, and the
reloaded artifact scored 1.0 on an 8-bit held-out lifetime.

This promotes verifier-gated staging and durable retention, not autonomous
program synthesis. Candidate proposal is explicitly supplied by the benchmark;
the deployed controller never receives `n_back`, correct actions, or hidden
verifier state. The verifier returns only receipt-linked scalar outcomes.

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_admission \
  --report-out /tmp/live-executive-admission.json
```
