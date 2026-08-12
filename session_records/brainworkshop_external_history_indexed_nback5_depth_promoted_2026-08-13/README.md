# Relative-age indexed external-history n-back-5 promotion (2026-08-13)

This record promotes the generic `history_indexed` external-compute reader.
The reader uses the memory-side relative age and presence mask to place valid
learned events in eight opaque age-addressed slots. It receives no family
name, target bit, correct action, or raw symbol format. The current event is a
separate learned tensor.

Configuration:

- family: `nback5`
- event reader: `history_indexed`
- event window size: `0` (no fixed window)
- query count: `0` (dynamic active history; five previous records plus current
  event once the lifetime reaches depth five)
- updates: `512`
- batch size: `32`
- attempted-outcome BCE with entropy weight `0.01`
- replayed examples: `0`

Both seeds reached `1.0000` on all four fresh lifetimes. The controller and
event frontend remained byte-identical. Missing-history, corrupted-history,
action-shuffled, reward-shuffled, and n-back-4 depth-shift controls all stayed
below the mastery gate.

This promotes bounded relative-age addressing and information preservation. It
does not establish learned unbounded compression, unrestricted memory growth,
arbitrary program induction, or general continual learning.

Reports:

- `report_seed17.json`
- `report_seed18.json`
