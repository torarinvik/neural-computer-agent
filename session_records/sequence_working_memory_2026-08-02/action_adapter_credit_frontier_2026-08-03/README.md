# Joint action-adapter credit arm (pre-registration, 2026-08-03)

The accepted frontier already shows that the newest skill residual can be
causally used, but lightweight routing and additional data did not improve
the child over its parent. This arm tests whether the frozen generic action
projection is the remaining credit bottleneck.

Only the existing final skill slot and the inherited generic action adapter
are trainable. The arm uses 128 fresh span-11 mixed lifetimes, 128 protected
span-10 and span-9 lifetimes, 128 protected blank span-11 lifetimes, 32
epochs, batch 512, learning rate 0.0005, binary outcome-complement and
critic losses, and 0.1 gate/logit protection. No new input, position window,
operation label, or semantic branch is added.

It must pass positive paired child-over-parent acquisition, newest-slot
causality, old retention, blank, and full-reset gates before any scaling.
