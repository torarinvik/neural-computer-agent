# Adversarial reversal and capacity pressure

This three-seed audit tests the learned reliability gate after two nonlinear
factored regimes are already committed. A low-error corruption is vetoed and
does not stage a candidate; a fresh gate-disabled control accepts the same
corruption. Clean returns to both historical regimes still route correctly
with the production gate active.

A third novel regime is refused while the two-slot capacity is full. Verified
retention growth expands capacity from two to three without changing committed
content, after which the novel candidate is isolated, promoted, and routed.
All three regimes remain addressable, the base/controller/context encoder stay
frozen, no old evidence is replayed, and persistence is exact.

This promotes bounded adversarial reversal and capacity-pressure behavior. It
does not establish unrestricted memory growth, multimodal grounding, arbitrary
new computation, or general continual learning.
