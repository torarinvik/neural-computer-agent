# Outcome-only external-history depth selection (2026-08-12)

This audit tests an isolated memory-side policy that chooses the active query
depth for each opaque external computation file. Candidate depths `1..5` are
probed in ascending order. A depth becomes preferred only after every probe
lifetime passes the stable-prefix mastery gate; if all candidates fail, the
policy exposes no depth. The policy stores only attempted query counts and
scalar verifier outcomes, is checksummed and reloadable, and cannot update the
controller, event frontend, or external file parameters.

The five-file bank is `symbol_parity`, `triplet_parity`, `parity2`,
`switch_binary`, and `nback4`. Across seeds 17 and 18, the policy selected the
same minimal stable profile `[1, 3, 2, 2, 5]`. Every selected depth retained
its file at `1.0000` on four fresh lifetimes. Controller/frontend immutability,
file immutability, exact policy reload, shuffled-outcome fail-closed, and zero
replay all passed. Calibration performed zero optimizer updates.

This promotes outcome-only active-depth selection as an external memory
contract. It does not establish neural depth inference, learned compression,
unrestricted memory growth, arbitrary program induction, or general
continual learning.
