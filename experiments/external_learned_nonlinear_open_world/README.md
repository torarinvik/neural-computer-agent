# Learned nonlinear open-world factual memory — rejected

This diagnostic pressure test replaces the promoted fixed random-feature
factual basis with a trainable `ExternalTransitionModel` MLP. The controller
and context encoder remain frozen. Four nonlinear regimes receive only `48/64`
training rows, and each current window is optimized through the router's new
`streaming_gradient` protocol without retaining raw provisional rows or
replaying old-regime evidence.

The learned MLP can often fit its current held-out regime, but it cannot yet
reliably route revisits. With a strict factual routing tolerance (`0.01`), all
three seeds either entered capacity instead of matching old slots or matched
only a subset of revisit windows. One seed also failed the stricter `0.08`
held-out quality gate. A loose routing tolerance was tested during smoke work
and prematurely matched novel regimes, so it was not promoted.

| seed | all acquisition quality | revisit routing | verdict |
| ---: | :---: | :---: | :--- |
| 82601 | pass | fail | reject |
| 82602 | fail | fail | reject |
| 82603 | pass | fail | reject |

The result rejects the naive substitution of a trainable MLP for the fixed
random-feature basis. It does not reject learned nonlinear factual models in
general. The next design must improve representation/routing stability or
meta-learn an online initialization before claiming this boundary.

The new `streaming_gradient` protocol is retained as a reusable primitive: it
updates caller-owned learned candidates from current evidence only, refuses
raw provisional-row retention, and persists independently from the frozen
controller. Current-window local updates are accounted separately from old
regime replay.
