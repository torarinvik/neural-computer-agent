# Behavior-verified int8 artifact quantization (2026-08-05)

This record pushes the seven-view external fallback artifact beyond float16
compression. Every floating tensor is quantized independently to symmetric
int8 with an explicit positive scale tensor. The artifact memory stores the
opaque quantized mapping and integrity hash; the caller-owned growth boundary
decompresses it before loading into the frozen controller.

## Promoted result

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| uncompressed tensor payload | 202,944 bytes | 202,944 bytes |
| int8+scale payload | 50,848 bytes | 50,848 bytes |
| payload ratio | 0.2506 | 0.2506 |
| uncompressed serialized file | 212,863 bytes | 212,863 bytes |
| quantized serialized file | 69,771 bytes | 69,771 bytes |
| serialized ratio | 0.3278 | 0.3278 |
| minimum quantized behavior | 0.7227 | 0.7148 |
| quantized behavior preserved | pass | pass |
| quantized wrong-view causality | pass | pass |
| aliases/reload/checksum/core | pass | pass |
| quantization optimizer updates / replay | 0 / 0 | 0 / 0 |

The seven-view route chain remained `1.000` and `0.998`; all prior-extension
attempt, shuffled-outcome, frozen-extension, and no-replay gates remained
true. Quantized behavior differed only within the predeclared five-point
retention tolerance from the uncompressed audit.

## Claim boundary

This promotes behavior-verified per-tensor int8 storage quantization for the
bounded seven-view artifact chain. It is not learned compression, arbitrary
new computation, open-ended memory growth, or general continual learning. The
codec is replaceable and deliberately isolated from the controller.

Reports are in `report_seed69316.json` and `report_seed69317.json`.
