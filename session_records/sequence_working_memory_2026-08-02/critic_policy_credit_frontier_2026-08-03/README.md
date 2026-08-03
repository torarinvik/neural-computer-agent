# Detached critic-to-policy credit arm (pre-registration, 2026-08-03)

The prior direct outcome update can create a positive newest-slot gap without
improving the child over its parent. This arm uses the existing temporary
action-conditioned critic as an explicit credit bridge: it is trained only
from the attempted opaque action and scalar outcome, then its detached
per-action success distribution distills into the existing successor slot on
fresh rows. The critic is discarded after training.

The parent is the accepted missing-evidence frontier. The arm uses 128 fresh
span-11 mixed lifetimes, 128 protected span-10 and span-9 lifetimes, 128
protected blank span-11 lifetimes, 32 epochs, batch 512, learning rate
0.0005, critic width 128, critic policy weight 1.0 after four warmup epochs,
binary complement losses, and 0.1 gate/logit protection.

Promotion requires positive paired child-over-parent acquisition, causal
newest-slot contribution, old retention, blank, and reset gates.
