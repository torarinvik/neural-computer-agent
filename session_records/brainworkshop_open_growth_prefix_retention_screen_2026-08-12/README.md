# Open external-compute growth with append-time prefix retention

This is a decisive negative frontier screen for the canonical replay-free
external-compute learner. The harness was strengthened so every append
transaction fresh-probes every previously admitted file. A candidate is
rolled back if its direct mastery fails or if any protected prefix file falls
below the stable threshold.

## Configuration

Seed `17`, four-event generic event window, eight scheduled files, eight
candidate attempts, 192 scalar-outcome optimizer updates per candidate, batch
size 32, four fresh retention lifetimes, 256 route updates per appended file,
and twelve transition batches. The controller and event encoder were frozen;
no verifier rows were replayed.

## Result

Seven files were admitted and all seven passed direct stable mastery. Every
append-time protected-prefix probe passed as well; the worst protected-prefix
fresh accuracy was `0.8551`. The eighth candidate, n-back-4, was rejected with
fresh accuracies `[0.7625, 0.7875, 0.7563, 0.7375]`. The failure is therefore
the bounded four-event representation, not catastrophic forgetting or prefix
corruption. The earlier four-event n-back-3 file remained at `1.0000` across
its append-time probes.

The matched reward-shuffled control stayed below `0.2761`. Routing, same-cue
reversal, old-file retention, reload, frozen-core, unchanged-file, and
zero-replay gates passed for the seven-file prefix. The overall run is
rejected because the requested eighth file was not admitted.

## Accounting

The run charged `1,197,568` primary training verifier bits, `33,792`
append-time protected-prefix verifier bits, `101,632` primary logical
lifetimes, `1,536` optimizer updates, and `1,628` route-memory updates. The
reported stable threshold is `null` because the target eight-file gate did
not pass. The raw process report was written to `/tmp/open-growth-8-seed17.json`;
the compact versioned summary is `report_summary.json`.

## Claim boundary

This validates append-time stable-prefix gating and bounded seven-file
external-compute growth. It does not establish unrestricted history,
capacity-free memory, arbitrary program induction, or general continual
learning. The next pressure rung is a scalable external history contract that
can make n-back-4 learnable without widening a fixed event window, followed by
replicated append-time retention screens.
