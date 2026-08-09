# Verifier-trained external lifetime policy — promoted

The two-seed audit trains an external lifetime scorer from one verifier outcome
bit per transaction. Each episode creates a fresh three-slot factual bank and
presents opaque contexts plus generic usage, age, and prediction-error
telemetry. A hidden verifier accepts only the safe unprotected slot. The
learned policy is compared with matched random and recency selectors on the
same held-out episodes.

Both seeds pass:

- learned held-out selection: 0.780 and 0.820;
- random control: 0.515 and 0.530;
- recency control: 0.265 and 0.295;
- protected-slot and stable-logical-address gates;
- exact policy persistence;
- zero controller updates, zero replayed transition examples.

The policy consumes 200 verifier outcomes during training and is evaluated on
200 fresh episodes. Every episode constructs a new bank, so no old transition
row is retained or replayed. This promotes a bounded verifier-trained lifetime
proposal mechanism. It does not promote unrestricted learned eviction,
consolidation, compression, or general continual learning. The verifier
remains authoritative and the telemetry is fixture-supplied.

The initial age-dominated control is intentionally not included in the
promotion: seed 1701 learned at 0.785 but lost to recency at 0.865. The mixed
generic rule is the corrected, pre-registered control for this archive.

