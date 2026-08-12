# Configurable relative-age indexed external-history n-back-16 promotion (2026-08-13)

This record promotes the explicit-width form of the generic `history_indexed`
external-compute reader. The replaceable file receives sixteen opaque
relative-age slots from memory-provided ages and a presence mask, plus the
current learned event. It receives no family name, target bit, correct action,
or raw symbol format.

Configuration:

- family: `nback16`
- event reader: `history_indexed`
- age-slot count: `16`
- event window size: `0` (no fixed window)
- query count: `16` (fifteen previous records plus current event)
- updates: `512`
- batch size: `32`
- attempted-outcome BCE with entropy weight `0.01`
- replayed examples: `0`

Across seeds `17` and `18`, all four fresh lifetimes reached `1.0000`. The
controller and event frontend remained byte-identical. Missing-history,
corrupted-history, action-shuffled, reward-shuffled, and n-back-8 depth-shift
controls all remained below the `0.80` mastery threshold.

This promotes a versioned, bounded 16-step relative-age address space. It does
not establish unrestricted history growth, learned compression, arbitrary
program induction, or general continual learning.

Reports:

- `report_seed17.json`
- `report_seed18.json`
- `sample_efficiency_ledger.json`
