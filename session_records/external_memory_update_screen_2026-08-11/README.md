# External-memory update screen

This screen tests whether replacing replay-free sufficient statistics with a
trainable nonlinear external transition memory is a higher-ROI route to
continual learning. The controller remains frozen, each update receives only
fresh staged evidence, and optimizer updates are counted separately.

The streaming-gradient arm used six target lifetimes on seeds `80, 90, 91,
93, 98`. It produced zero complete passes. Every run failed the independent
held-out model-family gate after `27` external-memory optimizer updates total,
with `240` transition rows consumed once, `195` unique verifier bits, and zero
replay. This rejects the arm at the current update schedule; it does not prove
that dynamic neural memory is impossible, only that a small number of online
gradient updates is not a substitute for a better evidence-routing and
credit-assignment mechanism.

A separate 24-seed width screen increased the replay-free random-feature basis
from 128 to 256. It fell from `14/24` to `13/24` complete passes, so capacity
alone is rejected as the next intervention.

Admission timing shows the same tradeoff. Staging after one row fell to
`8/24`; staging after three rows reached `12/24`; the six-row baseline reached
`14/24`. Early candidate writes therefore destabilize identity before the
promotion evidence is reliable.

Claim boundary: negative architecture-screening evidence, not a claim about
general neural external memory or general continual learning.
