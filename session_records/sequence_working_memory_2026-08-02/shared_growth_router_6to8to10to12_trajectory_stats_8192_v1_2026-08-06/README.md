# Shared candidate growth router: 6 → 8 → 10 → 12 at 8,192 updates (2026-08-06)

This is the promoted acquisition-efficiency rung for the shared,
permutation-equivariant external growth router. It uses the learned
trajectory-statistics query and random opaque candidate keys from the
16,384-update 32-capability audit, but halves each shared expansion from
16,384 to 8,192 optimizer updates.

The controller, context encoder, earlier route state, and earlier credit heads
remain frozen after each shift. New routes receive fresh paired scalar
outcomes only; no earlier examples are replayed.

## Promoted result

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| phase-1 minimum route selection (length 8) | 0.9844 | 0.9844 |
| phase-2 minimum route selection (length 10) | 0.9688 | 0.9063 |
| phase-3 minimum route selection (length 12) | 0.9219 | 0.9063 |
| old route / candidate permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| operational route permutation | 0.9875 | 0.9802 |
| causal credit across all shifts | passed | passed |
| full-bank protection / reversal / recovery | passed | passed |
| reward-shuffled false selections | 0 | 0 |
| replayed examples | 0 | 0 |

Both seeds pass old-route retention, direct and operational permutation,
causal new-route recovery, all-shift credit, full-bank protection, isolated
reversal and recovery, reward-shuffled null, and zero-replay gates. The bank
still reaches 32 opaque capabilities across three sequential shifts.

## Accounting

Per seed: 2,965,768 unique verifier bits, 56,840 unique logical lifetimes,
55,552 optimizer updates, 0 replayed examples, and 3 distribution shifts.
Across both seeds: 5,931,536 verifier bits, 113,680 logical lifetimes,
111,104 optimizer updates, and 0 replay. The prior promoted rung used 104,704
optimizer updates per seed, so this reduces total optimizer updates by 46.9%
while retaining every hard gate.

## Claim boundary

This promotes a more acquisition-efficient bounded shared-router growth rung.
It still does not establish unbounded growth, general program synthesis,
broad multimodal transfer, or general continual learning. The query remains
a fixed trajectory-statistics representation and candidate keys remain opaque
random rows whose associations are learned externally.
