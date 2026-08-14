# Live nonstationary executive-route promotion

This record audits a real route reversal inside the live tick runtime. Two
immutable, already verified executive skills were admitted: delay 1 in slot 0
and delay 2 in slot 1. A visible cue was encoded through the ordinary frozen
event frontend. The route ledger was calibrated to delay 1, then the private
verifier rule changed to 2-back behind the same cue. The controller, decoder,
and executive programs never trained.

After a checksummed `AgentBrain.bank` save/reload, the first two reversal
lifetimes correctly exposed failure of slot 0 (`0.5000`, `0.3750`). The route
policy then avoided the recently reversed slot, probed slot 1, and retained it:
the final three reversal lifetimes selected slot 1 and scored `1.0000` on all
8 eligible bits. The preferred order became `[1, 0]`.

The old slot remained intact: three forced slot-0 1-back lifetimes scored
`1.0000` on all 9 eligible bits. A cue-shuffled 2-back control selected the
untrained fallback slot 0 and scored `0.3750`. Bank reload was exact both before
and after reversal. The run used 27 calibration bits, 40 reversal bits, 27
forced-retention bits, and 8 control bits across 12 logical lifetimes. Route
updates were 9, optimizer/controller/decoder/executive-program updates were
zero, and replay was zero.

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.live_executive_route_reversal \
  --report-out /tmp/live-executive-route-reversal.json
```

This promotes bounded same-cue nonstationary routing, immutable skill
retention, and restart-safe route evidence. It does not claim autonomous
program induction or physical desktop deployment.
