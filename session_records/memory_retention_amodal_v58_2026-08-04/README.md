# v58 feedback-residual transfer audit

v58 adds a zero-initialized, protocol-agnostic feedback residual to the
memory value path. It preserves the parent and narrow retention gates, and all
three retained runs pass the per-run promotion gate. It improves fresh-parent
qualification, but the matched transfer population still fails: only seed 19
reaches a qualified transfer ratio; seed 17 fails transferred parent
qualification and seed 18 fails fresh-retention stability. No checkpoint or
population transfer claim is promoted.

Persistent reload, checksum rejection, and recovery pass for all three runs.
The unseen-token recall rates are `0.742`, `0.719`, and `0.754`, so the
feedback residual does not solve address generalization by itself.

The mechanism remains an opt-in training diagnostic pending a larger matched
transfer population and a versioned checkpoint decision.
