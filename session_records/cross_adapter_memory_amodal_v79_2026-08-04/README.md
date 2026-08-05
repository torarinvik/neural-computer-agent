# v79 capacity-one cross-adapter memory frontier

v79 fixes the event lifecycle used by the cross-adapter trainer. A probe now
previews an opaque action without advancing controller state; the payload is
inserted once, together with its scalar outcome. Previously the same payload
was inserted once for action sampling and again for outcome storage. With a
four-token event window, that duplicate insertion evicted the cue before all
three rows were processed, making random-position cue selection partially
unlearnable.

The corrected one-slot rung uses an explicit `memory_write_threshold=0.5`,
stable-content prior-event binding, and a trainer-only paired intervention
that isolates the suffix after a candidate write. These are generic event,
memory, and outcome-credit mechanisms; no target index, verifier bit, slot ID,
or semantic label enters the deployed controller.

At 512 steps, all three seeds pass the main writer/reader and causal-control
checks. Writer recall is `0.993/0.999/0.998`, reader recall is
`0.810/0.997/0.856`, clear/corrupt/swapped-slot controls remain near chance,
and persistent reload/recovery pass. Seed 17 also passes the fresh-token
population gate (`0.941` minimum reader recall; `+0.427` minimum aligned-vs-raw
gain). Seeds 18 and 19 remain below the strict fresh-token minimum (`0.776`
and `0.808`), and their 1024-step diagnostics improve those minima only to
`0.828` and `0.830`. Therefore capacity-one compression is not promoted as a
three-seed population capability yet.

The reward-shuffled control remains at chance (`0.504` writer and reader),
supporting causal use of verifier outcomes. The implementation breakthrough
is qualified: the prior failure was primarily self-inflicted event-window
eviction, not an inherent inability of the one-slot controller to bind and
retain a randomized target. The remaining bottleneck is fresh-token
generalization of the learned write policy.

Exact per-seed summaries and accounting are in `reports.json` and
`sample_efficiency_ledger.json`.
