# Routed generated composition medium rejection (2026-08-06)

Status: rejected; no weights or artifact promoted.

The explicit external composition router with two independent program slots
was trained for 256 updates on all six sampled two-primitive compositions. It
reached `0.6406` held-out behavior and never reached a stable `0.75` prefix.
The frozen parent was stable and unchanged; replay was zero.

This rejects learning the full grammar from scratch at the tested evidence
budget, not the external state boundary itself. The next repair is curriculum
expansion: acquire one composition, verify it, then add new compositions with
fresh outcomes and measure retention of the earlier composition without
replay.
