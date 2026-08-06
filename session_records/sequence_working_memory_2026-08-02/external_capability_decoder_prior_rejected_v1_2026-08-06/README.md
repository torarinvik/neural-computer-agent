# Retained decoder prior — 2026-08-06

Status: rejected as a promoted continual-learning strategy; retained as a
reusable-interface diagnostic.

The downstream head-only consumer was initialized with a copy of the retained
head decoder, while the head program, shared controller, and original head
decoder remained frozen. Raw events stayed hidden from downstream programs and
all consumer updates used fresh verifier examples with zero replay. The
default baseline used a fresh consumer decoder under the same pipeline.

| arm | consumer stable bits | target accuracy | blank accuracy | reward-shuffled accuracy |
| --- | ---: | ---: | ---: | ---: |
| retained decoder prior, full | 22,528 | 0.8398 | 0.5586 | 0.4375 |
| fresh decoder baseline, full | 16,384 | 0.8633 | 0.5586 | 0.4453 |

The prior produced a useful medium-rung endpoint (`0.7109` versus `0.6406`)
and all full-rung causal, persistence, frozen-core, and no-replay controls
passed. However, its early fluctuations delayed the first threshold prefix;
the stable-prefix gate therefore rejects it as a sample-efficiency gain. This
is evidence that output-interface reuse transfers a shortcut-like initial
alignment but does not yet provide stable continual learning. The retained
decoder remains a candidate for future calibrated or distillation-based
initialization, not an admitted default.

The sub-minute pilot stayed at chance and is included to record the required
curriculum ladder. Full accounting is in `sample_efficiency_ledger.json` and
the raw reports are kept beside it.
