# Verified external temporal-memory compaction

This two-seed promotion adds the missing commit boundary to the canonical
append-only content memory. Three learned records are written: an exact source
key, a nearby source alias with the same opaque capability address, and an
exact target key. A held-out route verifier approves only the source/alias
merge; a deliberately destructive source/target merge is rejected before it
can mutate the memory.

Seeds `17` and `18` both retained exact and related-key source and target
routes at `1.0000` after reducing three records to two. Reload and checksum
corruption controls passed, stale-version commits were rejected, and the
controller, event encoder, and frozen external capability file remained
unchanged. Each seed used `313,856` unique verifier bits, `33,024` logical
lifetimes, `512` optimizer updates, `520` route-memory updates, three memory
writes, and zero replay.

This promotes verifier-gated compaction of redundant learned content keys. It
does not establish arbitrary compression: distinct capabilities still require
distinct representational capacity. It also does not establish unrestricted
memory growth, arbitrary new computation, or general continual learning.

Reports are `seed-17.json` and `seed-18.json`. The experiment is implemented
in `experiments/brainworkshop_canonical/external_temporal_verified_compaction_growth.py`.
