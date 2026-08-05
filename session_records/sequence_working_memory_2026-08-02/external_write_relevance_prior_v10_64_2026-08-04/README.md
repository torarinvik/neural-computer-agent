# External writer v10 — seed 17, 64-update rung

Status: promoted narrow causal rung; no checkpoint promotion.

The controller is frozen during retention. A separately versioned external
memory writer receives opaque controller-native observations and learns from
three common-random write counterfactuals. The frozen relevance prior is
protected by a bounded `tanh` residual.

Key results:

- target-first: `0.949`
- target-last: `0.971`
- intact: `0.965`
- mastered-parent retention: `0.973`
- unseen-token minimum: `0.957`
- cue gain: `0.441`
- replayed examples: `0`

This qualifies the isolated overwrite-credit mechanism only. It does not claim
general continual learning, arbitrary new computation, or persistent-memory
transfer.
