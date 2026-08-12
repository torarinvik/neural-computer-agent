# Promoted compositional temporal reversal

This audit adds one nonstationary boundary to the promoted repeated online
compositional-growth stream. After three anonymous shared-basis pairs have
been acquired and consolidated, one composed external-memory row receives a
new opaque value regime under the same route keys. A separately trained
`OpaqueRegimeChangePolicy` must keep unchanged evidence as an exact no-op and
propose replacement for shifted evidence. A verifier-gated copy-on-write
transaction then replaces only that row while retaining the other composed
routes.

Across seeds `17`, `18`, and `19`, both forward and reversed insertion orders
passed every gate. The learned detector scored `1.0000` on stable-keep and
shift-replace in all three runs. It dominated the fresh control by never
regressing on either class and strictly improving at least one class. Stable
inputs left memory version and digest unchanged; shifted inputs triggered
replacement, which passed the route verifier and reloaded exactly. All three
composed routes remained available, corruption was rejected, the controller,
event encoder, and acquired temporal file stayed byte-identical, and replay
was zero.

Per seed the audit consumed `43,904` unique temporal verifier bits plus
`1,000` detector utility bits, `49,112` logical lifetimes, `4,000` optimizer
updates, and zero replayed examples. The detector is trained externally; the
controller and event encoder are frozen. The changed artifact is an external
memory-regime replacement, not evidence that the frozen controller learned a
new task or that the system has general continual learning.

This promotes a narrow learned no-op/replace boundary for verifier-gated
external memory. It does not qualify unrestricted memory growth, arbitrary
new computation, universal regime discovery, or general continual learning.
Raw reports are `seed-17.json`, `seed-18.json`, and `seed-19.json`.
