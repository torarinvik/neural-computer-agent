# Verifier-gated shared-basis regime replacement — promoted narrow boundary

Status: `PROMOTED` as a protected-scope external memory replacement result.

This rung adds `shared_basis_rewrite_v1`: a copy-on-write candidate can change
the logical row set of one external scope while preserving other scopes. The
rewrite is independently retention-verified, expected-version checked, and
atomically persisted. The existing representation-only compression API is
unchanged.

The memory has a protected six-route source scope and a twelve-route working
scope. Initially, the working scope contains two incompatible rank-two
subspaces, so the global structure is rank six. The external v2 policy selects
rank `8` for the old regime. The working scope is then replaced by twelve new
routes in a different rank-two subspace; the protected scope remains, the old
working addresses are removed, and the policy selects rank `4` for the new
global regime.

| seed | held-out rank 2/4/8 | fresh rank 2/4/8 | old/new choices | final physical/dense value scalars |
| ---: | --- | --- | --- | ---: |
| 17 | `0.9688/1.0000/1.0000` | `0.9688/0.0000/0.0000` | `8 → 4` | `136/288` |
| 18 | `0.9688/1.0000/1.0000` | `0.9688/0.0000/0.0000` | `8 → 4` | `136/288` |

Both seeds passed protected-route retention, new-route admission, old-route
removal, forward/reversed runs, exact reload, stale-version rejection,
checksum-corruption rejection, frozen controller/encoder, and zero replay.

This promotes bounded verifier-gated regime replacement and capacity reuse. It
does not establish autonomous change detection, arbitrary semantic regime
discovery, unrestricted memory growth, arbitrary new computation, or general
continual learning. The next pressure is learned replacement timing under
unknown regime boundaries, repeated alternating replacements, and capacity
pressure with more than one protected capability.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `sample_efficiency_ledger.json`
