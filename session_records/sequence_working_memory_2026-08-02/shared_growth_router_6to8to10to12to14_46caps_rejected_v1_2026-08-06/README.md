# Shared growth router: 46-capability frontier rejection (2026-08-06)

This is the next pressure test after the promoted 32-capability 6→8→10→12
rung. It adds a fourth length-14 shift, producing 46 total capabilities, and
keeps the controller, prior route state, and earlier credit state frozen.

## Rejection evidence

| control | final route floor | retention | result |
| --- | ---: | --- | --- |
| hidden 256, 8,192 updates per shift | 0.8125 | failed | rejected |
| hidden 256, 8,192/8,192/8,192/12,288 | 0.7188 | failed | rejected |
| hidden 512, 8,192 updates per shift | 0.7969 | failed | rejected |

All three controls retain old routes, candidate permutation, causal credit,
reward-shuffled null, and zero replay. The hard failure is late-shift
acquisition: one or more length-14 routes remain below the stable mastery
threshold, so the full-bank retention ledger refuses protection and eviction
cannot safely proceed. Increasing width or adding late-shift updates does not
repair it; the adaptive extra-update control is worse.

## Accounting

The hidden-256 fixed-budget control used 4,292,984 unique verifier bits,
75,128 logical lifetimes, and 73,728 optimizer updates. The adaptive control
used 4,555,128 verifier bits, 83,320 logical lifetimes, and 81,920 optimizer
updates. The hidden-512 control used the same verifier and optimizer counts as
the fixed-budget control. All controls replayed zero examples.

## Confidence-aware staged admission audit

The fixed 8,192-update control was replicated with seeds 69316 and 69317
through the memory-side staging boundary. This audit consumed the existing
scalar route outcomes; it added no verifier bits, optimizer updates, or replay.

| seed | stable candidates admitted | candidates pending | occupied rows | protected occupied rows |
| --- | ---: | ---: | ---: | ---: |
| 69316 | 43/46 | 3 | 43 | 43 |
| 69317 | 39/46 | 7 | 39 | 39 |

Every occupied row was protected, and every candidate below the stable-prefix
gate remained outside executable memory. Pending candidates therefore consumed
no executable capacity and could not evict or dilute a protected row. This is
a lifecycle-safety result, not a route-learning promotion: the 46-capability
frontier remains rejected because the late length-14 routes do not all earn
stable mastery.

The staged reports are `report_seed69316_staged_admission.json` and
`report_seed69317_staged_admission.json`.

## Claim boundary

This rejects a naive extension of the 32-capability rung, not the shared-router
architecture itself. The next bottleneck is confidence-aware late-shift route
acquisition and capacity planning: a route must earn stable protection before
the memory bank can claim open-ended growth.
