# External online transition-context routing

This pressure test moves learned context formation from a finite held-out
bundle toward online continual use. A frozen controller is paired with an
external transition-model bank and a learned context encoder. Each incoming
opaque transition is routed by factual one-step prediction error. Unmatched
rows remain provisional; after a current-stream bundle is complete, the
encoder forms an opaque key and the bank admits an isolated slot.

The stream alternates already-mastered regimes, discovers a novel regime
without a regime label, returns to old regimes, and then attempts a fourth
regime at a full capacity boundary. The report accounts for current-stream
replay separately from old-prior replay and includes routing, retention,
frozen-controller, persistence, and no-growth controls.

This is still bounded online identity, not general continual learning: the
context encoder is pretrained, the admission window is finite, and capacity
pressure is refused rather than solved by consolidation or compression.

```text
.venv/bin/python experiments/external_online_transition_context/train.py \
  --seed 70011 \
  --report-out /tmp/online-transition-context.json
```
