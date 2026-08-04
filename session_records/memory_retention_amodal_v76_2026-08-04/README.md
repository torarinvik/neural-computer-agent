# v76 promoted three-slot / two-row retention

v76 is the corrected three-slot/two-row outcome-only retention qualification.
It uses the canonical v27 controller/runtime, a bounded two-row content-
addressed memory, a four-token event window, balanced target positions,
counterfactual write credit, randomized opaque event-token blocks, parent
rehearsal, and persistent-memory audits.

The implementation change is protocol-agnostic. The controller now combines
the learned memory address with a residual learned-event identity path, and
the write policy receives the strongest prior-event binding signal rather than
only an average prior match. This prevents a near-collision distractor from
displacing the cued row while keeping the address learned and replaceable.

All three seeds pass the pre-registered gate. Intact recall is
`0.9238/0.9971/0.9990`, target-first recall is `0.9980/0.9971/0.9980`, and
target-last recall is `0.8457/0.9990/0.9971`. The minimum unseen-token
population recall is `0.9141/0.8359/0.9297`. Persistent reload recall is
`0.9336/0.9922/0.9961`, recovery is `0.9375/1.0000/1.0000`, and checksum
corruption is rejected for every seed.

This promotes only learned three-slot/two-row retention with bounded
persistent memory under the outcome-only verifier. It is not evidence of
natural-language, speech, physical-action, or general episodic-memory
capability. Cross-adapter synthetic retrieval remains separately qualified in
v75.

The compact per-seed accounting records and aggregate sample-efficiency ledger
are in this directory. Raw reports were retained as disposable local
diagnostics outside Git.
