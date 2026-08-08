# Continual shared-blueprint anchor — rejected

This control allowed shared register-operator weights to update on later
source primitives while freezing old opaque instruction codes. A quadratic
anchor penalty to the previous shared weights was applied using only fresh
outcomes for the current primitive.

Anchor weights `1.0` and `10.0` were tested across seeds `69316` and `69317`.
The new `rotate` target often mastered quickly, including `4,096` inherited
bits versus `8,192` fresh in one run. However, source primitives fell below
the hard mastery floor before target acquisition in every run. Safety controls
passed, but source retention rejected the intervention.

Conclusion: naïve whole-blueprint online updates are unsafe. Future
operator-family learning needs isolated trainable meta-state or protected
subspaces, not a scalar weight anchor.
