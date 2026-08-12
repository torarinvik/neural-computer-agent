# Shared-basis external-memory compression — promoted narrow boundary

Status: `PROMOTED` as a verifier-gated storage-contract result.

This two-seed canonical audit stores twelve distinct opaque values under
independent learned event keys in `PersistentSharedBasisContentAddressedMemory`.
The values share a two-dimensional external structure plus a small residual.
The controller and learned event encoder remain frozen. A rank-one
copy-on-write candidate is rejected by the held-out route/value verifier; a
rank-two candidate is committed only after all twelve logical routes remain
usable.

| seed | physical value scalars | after rank-2 | rank-1 error | rank-2 error |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 336 | 56 | 0.834600 | 0.004719 |
| 18 | 336 | 56 | 0.681139 | 0.005543 |

Both seeds passed forward and reversed physical row-order runs, independent
route retention, non-mutating rejection, stale-version rejection, exact
reload, checksum-corruption rejection, frozen controller/encoder, and zero
replayed examples.

The reduction is from `336` stored basis/coefficient scalars to `56`, while
the dense logical payload would require `192` value scalars. Logical record
count remains `12` throughout. The rank choice is deterministic SVD in this
first contract audit; no learned compression capability is claimed.

This promotes safe shared-structure representation in the replaceable memory
boundary. It does not establish learned rank selection, semantic equivalence
discovery, arbitrary new computation, unrestricted memory growth, or general
continual learning. The next pressure is an outcome-trained, verifier-gated
rank/structure proposal on evolving values, including residual growth and
long nonstationary retention streams.

Reports:

- `seed-17.json`
- `seed-18.json`
