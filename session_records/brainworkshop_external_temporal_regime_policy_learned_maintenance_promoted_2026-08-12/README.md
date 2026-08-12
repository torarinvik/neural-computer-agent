# Learned full-bank maintenance — promoted boundary

Status: `PROMOTED` as a bounded learned victim-selection result.

`ExternalCapabilityEvictionPolicy` learns which opaque candidate slot is most
disposable from one scalar verifier utility per fresh candidate bank. It sees
only generic candidate features: an opaque binding key, reliability telemetry,
and age telemetry. Candidate order is independently permuted. It does not see
semantic names, task labels, physical slot indices, or verifier targets.

| seed | held-out selector | fresh selector | forward victim | reverse victim | sibling partial / new partial |
| ---: | ---: | ---: | ---: | ---: | --- |
| 17 | `0.9648` | `0.3477` | correct | correct | `0.8047 / 0.8047` |
| 18 | `0.9258` | `0.3516` | correct | correct | `0.8438 / 0.8672` |

The selected weak slot is replaced through the residual bank's copy-on-write
verifier. The sibling capability remains retained, the new binding is then
learned and frozen, and controller/encoder updates plus replay remain zero.
This closes the hand-selected-victim gap for a narrow full-bank transaction.

This does not establish autonomous redundancy discovery, universal eviction
economics, arbitrary memory growth, or general continual learning. The next
pressure is long nonstationary maintenance with competing quality signals,
protected slots, and learned compression/eviction tradeoffs.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `sample_efficiency_ledger.json`
