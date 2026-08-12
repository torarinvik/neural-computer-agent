# Alternating hidden regimes — promoted bounded boundary

Status: `PROMOTED` as a replay-free repeated-reversal and capacity-reuse
result.

The frozen controller and event encoder are paired with an external learned
regime trigger and shared-basis memory. The live stream alternates opaque
working regimes `A → B → A → B → A → B` across five hidden boundaries. Before
each boundary, an identical incoming bank must be an exact keep/no-op. The
shifted bank must trigger a verifier-gated rewrite of one working scope.

Three protected scopes remain isolated and readable through every rewrite.
Each replacement uses new opaque addresses, so stale working routes must be
absent. The logical record count stays at `26` and factorized physical value
storage stays at `168` scalars versus `416` dense scalars across all six
checkpoints; replacement reuses capacity rather than appending old regimes.

| seed | trained stable keep / shift replace | fresh stable keep / shift replace | forward/reverse boundaries | physical history |
| ---: | --- | --- | ---: | --- |
| 17 | `1.0000/1.0000` | `1.0000/0.0000` | `5/5` | `168,168,168,168,168,168` |
| 18 | `1.0000/1.0000` | `0.9922/0.0000` | `5/5` | `168,168,168,168,168,168` |

Both seeds passed protected-scope retention, stale-route removal, exact
stable no-op version/digest checks, rank-four replacement and compression,
reload, stale-version rejection, checksum corruption, frozen controller and
encoder, and zero replay. This promotes a narrow repeated external-memory
boundary. It does not establish autonomous semantic change-point discovery,
unrestricted memory growth, arbitrary new computation, or general continual
learning.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `sample_efficiency_ledger.json`
