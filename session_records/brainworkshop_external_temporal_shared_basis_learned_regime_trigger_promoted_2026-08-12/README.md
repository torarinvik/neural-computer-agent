# Learned external regime trigger — promoted narrow boundary

Status: `PROMOTED` as a raw-value trigger-transfer result.

`OpaqueRegimeChangePolicy` compares an opaque current working bank with an
incoming bank using permutation-invariant spectral and cross-bank structure
features. It emits only `keep` or `replace` and learns from one scalar
verifier utility per pair. It receives no regime ID, task label, candidate
reconstruction error, or semantic metadata.

The canonical stream first presents stable evidence from the current regime.
The detector must make this an exact no-op: memory bytes and store version must
not change. A shifted bank then arrives; only after the detector proposes
replacement does the independent shared-basis verifier authorize the protected
scope-preserving rewrite.

| seed | trained stable keep / shift replace | fresh stable keep / shift replace | stable live action | shifted live action |
| ---: | --- | --- | --- | --- |
| 17 | `1.0000/1.0000` | `0.0000/1.0000` | keep | replace |
| 18 | `1.0000/1.0000` | `1.0000/0.0000` | keep | replace |

Both seeds passed stable no-op byte/version stability, automatic shift
detection, protected-route retention, old-route removal, new-route admission,
reload, checksum corruption, frozen controller/encoder, and zero replay. The
detector and structure policy each used only 1,000 unique scalar-utility
updates.

This promotes learned bounded replacement timing, not autonomous semantic
change-point discovery, unrestricted memory growth, arbitrary new computation,
or general continual learning. The next pressure is repeated alternating
regimes with unknown boundaries, multiple protected scopes, and capacity
pressure.

Reports and accounting:

- `seed-17.json`
- `seed-18.json`
- `sample_efficiency_ledger.json`
