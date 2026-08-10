# Outcome-trained stream-binding lifecycle policy

This archive records the next promoted boundary after open-set quarantine:
an external learned policy proposes which provisional identity should replace
which live identity. The frozen controller and event encoder remain outside
the policy. The policy sees only opaque prototypes and generic lifecycle
telemetry: observation counts, verifier reliability, delay statistics, and
prototype similarity.

Each proposal records its exact logging propensity and consumes one scalar
verifier outcome without replay. A proposal cannot mutate memory directly.
`ExternalOnlineStreamBindingMemory.replace_verified_track_with_provisional`
performs the live-track retirement and provisional admission as one
copy-on-write transaction, committing only after the verifier accepts the
candidate. A hold proposal is available for contradictory evidence.

Across seeds `2401` and `2402`:

- five anonymous streams were represented by two live and three simultaneous
  provisional identities;
- the trained policy achieved `1.0` safe-replacement accuracy and `1.0`
  contradiction/hold accuracy on both seeds;
- matched fresh policies scored `0.125` and `0.1667`;
- outcome-shuffled controls scored `0.125` and `0.2083`;
- propensity logging, atomic rejection, exact policy persistence, frozen
  encoder/controller, and zero-replay gates all passed;
- each seed used 960 unique verifier bits, 960 policy optimizer updates, and
  zero controller optimizer updates.

The raw reports are `report_seed2401.json` and `report_seed2402.json`.
`sample_efficiency_ledger.json` records the two-seed accounting and
`SHA256SUMS` protects this archive.

## Claim boundary

This promotes a narrow outcome-trained lifecycle proposal mechanism and atomic
retention safety. It does not establish learned verifier design, autonomous
eviction economics, unrestricted memory growth, arbitrary identity discovery,
or general continual learning. The next pressure is coupling the policy to
held-out factual model retention under delayed contradiction and real drift,
then measuring reduction in verified experience per newly retained capability.
