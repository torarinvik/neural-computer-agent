# Four-file external compute route bank — promoted

This record promotes the four-file outcome-only route-bank rung from
`experiments/brainworkshop_canonical/external_compute_route_bank.py`.

The run acquires four independent opaque external compute files behind one
frozen controller, event frontend, and generic register interpreter:

1. `symbol_parity`
2. `triplet_parity`
3. `parity2`
4. `switch_binary`

Each file is trained in isolation from fresh rendered lifetimes. After the
first file is mastered, later files are appended and trained without replaying
earlier examples or updating earlier file parameters. A
`PersistentOpaqueContextRouteEvidence` table learns the file address from a
learned event key and terminal scalar episode accuracy only.

Seeds `17` and `18` both pass:

- all four direct files remain stably mastered across four held-out lifetimes;
- all four learned contexts select the correct file at `1.0000` fraction and
  retain mastery after the full bank is grown;
- an unseen context conservatively selects the oldest file and remains near
  chance (`0.5096` and `0.5072` accuracy);
- a no-file control cannot reach the newest task (`0.4952` and `0.4790`);
- route state reload is exact;
- every file digest is unchanged after later growth;
- controller and event frontend digests are unchanged;
- replayed examples are zero.

Each seed uses `605,696` unique verifier bits, `49,152` unique logical
lifetimes, `768` optimizer updates, and `776` route-memory updates. The
promotion threshold is the first prefix that remains satisfied across all four
retention lifetimes.

## Measurement correction

The earlier four-symbol `switch` probe was not a valid 50% chance control:
`current != previous` is true about 75% of the time over four symbols. Its
approximately `0.75` score was therefore majority-class behavior, not evidence
of learned temporal reasoning. `switch_binary` uses a two-symbol rendered
alphabet, making the target balanced at 50%. The old probe is retained only as
a diagnostic and is not promotion evidence.

This promotes bounded append-only external-file routing, not unrestricted
memory growth, arbitrary program induction, or general continual learning.
