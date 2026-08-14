# Live opaque executive-route promotion

This record promotes a bounded multi-skill live route. Two already verified
temporal-equality executive artifacts (delay 1 and delay 2) were admitted to a
fresh `AgentBrain.bank`. A public mode cue was rendered through the ordinary
frozen `BrainWorkshopEventEncoder`; the route machine saw only its first learned
event tensor and scalar verifier outcomes. It never received `n_back`, a task
identifier, a symbol, or a correct action.

The run alternated 16 training lifetimes (8 per cue) for 136 unique eligible
verifier bits. The route ledger used one mean outcome per lifetime, preventing a
lucky action streak from promoting a partially correct skill. It learned
preferred orders `[0, 1]` for the 1-back cue and `[1, 0]` for the 2-back cue.
Three held-out lifetimes per cue then selected slots 0 and 1 respectively and
scored `1.0000` on every 9-bit and 8-bit lifetime. Route updates were 22, with
zero controller updates, zero decoder updates, zero executive-program updates,
and zero replay.
The measured wall time was 0.143 seconds on CPU; held-out live-tick p50/p99
latencies were approximately 0.435/0.612 ms.

This is a promotion of bounded opaque route execution, not open-ended route
discovery. The banked skills are pre-verified, the cue is visible, and the
context encoder is still a replaceable event adapter. It does not claim
autonomous program induction, physical GUI deployment, or generalization to an
unseen cue. Action-level rewards still travel through the normal exact-once
live outcome path; only selector mastery is lifetime-aggregated by default.

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_route \
  --report-out /tmp/live-executive-route.json
```

The canonical report is `report.json`.
