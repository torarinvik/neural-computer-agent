# Live bank-fed executive composition

This session demonstrates the next growth step after verifier-gated admission:
a new skill is derived from two already-admitted `AgentBrain.bank` slots rather
than receiving a new candidate artifact from outside the bank.

Parent slot 0 is a finite learned-event prelude and parent slot 1 is the
verified delay-2 temporal loop. The composed child uses explicit compatible
operator sharing and `final_emit_only` handoff semantics. The prelude processes
the same learned event but does not create an extra external action, so the
temporal operator sees an unshifted live history.

The child reached 1.0000 on all three live lifetimes (24 unique verifier bits),
was admitted into slot 2 with parent-slot and parent-digest provenance, and
retained 1.0000 on an 8-bit lifetime after exact bank save/reload. A matched
delay-1 parent produced `0.50`, `0.375`, and `0.50`; its composed child was
rejected and its bank digest was unchanged. Controller, decoder, and executive
program updates were zero; replay was zero.

This promotes bounded bank-fed composition and event-continuous execution, not
autonomous open-ended program synthesis. Parent-slot proposal is deterministic
and supplied by the benchmark; the controller never receives `n_back`, correct
actions, or verifier-private state. The verifier returns only receipt-linked
scalar outcomes.

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_composition \
  --report-out /tmp/live-executive-composition.json
```
