# Physical 2-cell Position 1-Back program rejection (2026-08-14)

Status: **frozen-controller mechanism retained; external program file rejected**.

This was the first promoted physical campaign with the task-learning ownership
boundary enforced. Brain Workshop ran ordinary Position 1-Back timing and
scoring, but its public stimulus distribution was restricted to two visible
grid cells. The learner received only display-captured RGB events, emitted the
ordinary position-match key, and learned from visible green/red/blue feedback.

The controller/executor digest remained
`a8bdac7ffafaa44b147ac9fb8f0aaf79982c3601a0418f6bdd833f403b5312f4`.
The external program digest changed from
`4bf66088f06774dc015d181625a3d8b7c2d5ddc3b1fcc316578cd9c3ffd5cc8e`
to `1e5dffbfe35b375a741607137376587116a88b449a869bf6d20c8245eebc5481`.
Thus 99 unique public outcomes caused exactly 99 program-file updates, zero
controller optimizer updates, and zero replay.

The curve was not stable. Session accuracy rose as high as `0.7000`, the
rolling score briefly reached `0.6250`, then fell; the final 32-outcome score
was `0.5000` and cumulative score was `0.4949`. This is far below the `0.80`
mastery gate, so neither the program weights nor a cell-count promotion are
retained. The next experiment stays at two cells and must causally improve or
diagnose the program learner before another roughly three-minute run.

The executor used here is a parameter-free live contract for history, causal
receipts, and program execution. This run does not claim that the intended
meta-trained general controller has already been produced.
