# Three-step external view growth

This audit extends the promoted two-step boundary by adding `rotate`,
`complement_rotate`, and `adjacent_xor` sequentially after the four-view
router is frozen. Each new view is trained from fresh paired scalar outcomes;
earlier extensions are frozen and their route examples are never replayed.

The three new views are compacted into one physical artifact row. A later
procedure must pass through the old route and each earlier extension as failed
opaque attempts before its own extension is opened. The report separately
checks old-route retention, cumulative routing, candidate permutation, prior
extension attempts, reward-shuffled controls, reload/checksum integrity, and
behavioral wrong-view causality.

This is a pressure test for a longer but still bounded external fallback chain.
It is not a claim of unrestricted memory growth, arbitrary new computation, or
general continual learning.

The same audit also creates a transactionally verified float16 representation
of the complete seven-view payload. The compressed artifact is reloaded and
executed through an explicit dtype-cast boundary; raw payload size and
behavioral retention are recorded in the report. This is caller-owned storage
compression, not a learned reasoning module.

The audit also evaluates per-tensor symmetric int8 quantization with explicit
scale entries. It is promoted only when decompressed behavior remains causal,
reloadable, and within the retention tolerance; the quantized representation
is never passed directly into the controller.

It also evaluates packed signed-int4 quantization with per-output-row scales
and explicit shapes. Two int4 values are stored per byte and decompressed
before controller loading. The representation is promoted only when behavior,
wrong-view causality, exact reload, corruption rejection, and frozen-core
gates pass across both seeds; this is a storage result, not learned
compression or general continual learning.

## Retention-aware three-step extension

The harness now applies the retention transaction to all three additions and
to each float16, int8, and packed-int4 representation. A candidate is probed
on fresh outcomes and behavior-verified before any protected source row can be
replaced. At the promoted acquisition budget, seed `69316` passes seven-view
growth and all three storage transactions, and seed `69317` independently
passes the same gates. The corrected three-step result is promoted across
seeds: route accuracy is `1.0000` and `0.9980`, with zero replay and frozen
controller/earlier extensions.

The earlier seed-`69317` rejection is retained as a historical control. It
exposed inconsistent accounting: retained floors used the raw minimum probe
outcome while the gate used the stable cumulative-prefix minimum. The harness
now uses the stable-prefix definition consistently and records a paired
full-precision source control. The storage codecs remain caller-owned,
behavior-verified representations rather than learned computation.
