# Behavior-verified packed int4 artifact quantization (2026-08-05)

This record pushes the seven-view external fallback artifact beyond float16
compression and per-tensor int8 quantization. The caller-owned codec stores
signed int4 values as packed nibbles, with an explicit per-output-row scale
and original-shape entry for every floating tensor. Decompression happens
before the strict frozen-growth loader; the controller and memory backend
remain unchanged and opaque to the codec.

## Promoted result

| Metric | Seed 69316 | Seed 69317 |
| --- | ---: | ---: |
| uncompressed tensor payload | 202,944 bytes | 202,944 bytes |
| packed int4 + scales/shapes payload | 30,184 bytes | 30,184 bytes |
| payload ratio | 0.1487 | 0.1487 |
| uncompressed serialized file | 212,863 bytes | 212,863 bytes |
| packed serialized file | 58,007 bytes | 58,007 bytes |
| serialized ratio | 0.2725 | 0.2725 |
| minimum packed behavior | 0.7227 | 0.7305 |
| three-step route accuracy | 1.0000 | 0.9980 |
| packed behavior preserved | pass | pass |
| packed wrong-view causality | pass | pass |
| aliases/reload/checksum/core | pass | pass |
| packed quantization optimizer updates / replay | 0 / 0 | 0 / 0 |

The complete seven-view route chain and candidate-permutation control both
passed. Every prior-extension attempt remained present, reward-shuffled new
view selection stayed at zero, and all packed-artifact integrity and frozen
core gates passed on both seeds.

## Claim boundary

This promotes behavior-verified packed per-output-row int4 storage
quantization for the bounded seven-view external artifact chain. It is not
learned compression, arbitrary new computation, open-ended memory growth, or
general continual learning. The codec is replaceable, caller-owned, and
isolated from the controller.

Reports are in `report_seed69316.json` and `report_seed69317.json`.
