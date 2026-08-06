# Retention-safe online executable-view growth v2 (2026-08-06)

This milestone tests whether a frozen controller can acquire one genuinely new
executable capability online while retaining four previously mastered routes.
The four old views are `forward`, `reverse`, `complement`, and
`complement_reverse`; the new view is `rotate`. All five are opaque external
memory views of one physical artifact row. The controller core and the
four-view router remain frozen.

The memory transaction is retention-safe. Fresh verifier probes first protect
the four old capabilities. The fifth artifact is constructed in a disposable
candidate store, probed eight times, and admitted only when its minimum outcome
is at least `0.70`. An independent behavior verifier also requires every old
and new operation to remain within `0.05` of its baseline. The adopted row is
protected after the transaction. Source memory is not mutated during candidate
evaluation, and route acquisition after extension uses zero replayed examples.

## Promoted evidence

Seeds `69316` and `69317` both passed every gate:

- old capability protection before extension and replacement protection after;
- stable new-candidate retention, with candidate floors `0.8945` and `0.8672`;
- five opaque views in one physical row;
- learned new-view route accuracy `1.000` on both seeds;
- combined five-view route accuracy `1.000` and `0.9844`;
- old-route retention accuracy `1.000` and `0.9688`;
- candidate permutation, wrong-view causal, reward-shuffle, reload, checksum
  corruption, frozen-core, frozen-router, and zero-replay controls.

Per seed, the accounting was `393,216` unique verifier bits, `98,304`
logical lifetimes, `6,144` optimizer updates, `40` retention observations,
and `0` replayed examples. Wall time was approximately `306.5` and `312.7`
seconds. The full reports are `report_seed69316.json` and
`report_seed69317.json`.

## Rejected controls

The default short budget (`64` old-artifact updates and `64` new-artifact
updates) was rejected because the new candidate did not establish stable
mastery at the `0.70` retention threshold. A deeper diagnostic with only `256`
new-artifact updates made the failure explicit: the old retained floors were
`[0.6445, 0.6055, 0.6484, 0.6367]`, while the new candidate outcomes ranged
from `0.6797` to `0.6875`. Neither the old nor new state met the declared
retention contract, so it was not promoted.

## Claim boundary

This promotes bounded, retention-safe online addition of one executable
external-memory view. It does not establish unrestricted memory growth,
learned byte compression, arbitrary program induction, or general continual
learning. The next bottleneck is scaling the same transaction and learned
address-acquisition contract across multiple independently acquired additions
under memory pressure, reversals, and held-out transfer.
