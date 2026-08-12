# Mixed-depth external-history n-back-5 retention and routing (2026-08-13)

This audit extends the promoted mixed-depth external-history boundary from
n-back-4 to n-back-5. One frozen controller and event frontend feed five
independently trained opaque external computation files. The active history
profile is `(4, 4, 4, 4, 6)`: the first four files receive three preceding
records plus the current event, while the n-back-5 file receives five
preceding records plus the current event. The complete lifetime remains in
append-only external history, and the shared event window is six records.

The files are `symbol_parity`, `triplet_parity`, `parity2`, `switch_binary`,
and `nback5`. Each file was trained for 512 optimizer updates from attempted
action outcomes only, using the same generic flattened-window basis. The
controller, event frontend, and previously acquired file parameters were
frozen while later files were learned. Route memory received only scalar
episode outcomes; replayed examples were zero.

Across seeds 17 and 18, every direct file and every routed file reached
`1.0000` on four fresh lifetimes. Every route selected the correct file at
`1.0000`; the serialized route table reloaded exactly; unknown contexts fell
back to the oldest file while remaining near chance (`0.5195` and `0.5094`);
the no-file controls were also near chance (`0.4859` and `0.4828`); all file
digests remained unchanged during routing; and the controller and event
frontend remained frozen.

This promotes replicated replay-free retention and routing at a deeper,
mixed bounded temporal profile. It does not establish learned depth selection,
learned compression, unrestricted memory growth, arbitrary program induction,
or general continual learning. The complete reports and accounting are in
`seed17_report.json`, `seed18_report.json`, and `sample_efficiency_ledger.json`.
