# Generic write-cost mini-rung

This parent-stable mini-rung tested the existing generic write-cost
regularizer after the v32 write-policy saturation diagnosis. Both arms used
the current v23 runtime, balanced-order retention, 512 requested parent steps
(the parent audit stopped after 288 effective updates), 256 retention steps,
and 64 alternating warmup steps.

The zero-cost arm passed the per-run retention gate, but it is not a population
promotion and no weights are curated. The `0.02` write-cost arm reduced the
write rate but failed the cue-gain gate and performed worse overall. This is a
negative intervention: penalizing writes alone does not solve outcome-only
credit assignment and can produce a different position shortcut.

Full reports and accounting are in the two JSON files in this directory. The
next intervention should target credit variance or memory-state observability,
with one change at a time.
