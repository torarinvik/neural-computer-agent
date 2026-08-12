# Routed nonstationary maintenance residuals — promoted boundary

Status: `PROMOTED` as a bounded multi-objective maintenance result.

The frozen external maintenance scorer is trained on reliability-dominated
victim selection. Two opaque context keys then route isolated residual
maintenance slots: one learns reliability selection and the other learns age
selection. Each slot receives only fresh scalar verifier utilities and is
activated/frozen after promotion. Unknown contexts fall back to the base
scorer. Candidate order is evaluated both forward and reversed.

| seed | reliability forward/reverse | age forward/reverse | unknown-key fallback | slot-A retained |
| ---: | --- | --- | --- | --- |
| 17 | `0.8828/0.8828` | `0.8711/0.8711` | `0.9844/0.9844` | yes |
| 18 | `0.8438/0.8438` | `0.8867/0.8867` | `0.9570/0.9570` | yes |

The 3,000-update base and 512 fresh updates per residual slot use zero replay;
the controller and event encoder remain frozen. This demonstrates isolated
adaptation to changing maintenance objectives while preserving an unknown-
context fallback.

This does not establish autonomous context-key discovery, universal
maintenance economics, unrestricted slot growth, or general continual
learning. The next pressure is repeated long-horizon allocation/replacement
with changing objectives and learned key/lifecycle management.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `sample_efficiency_ledger.json`
