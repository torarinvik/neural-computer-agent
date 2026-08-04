# Outcome value-baseline diagnostic rejection

v50 tests a training-only learned value baseline for scalar parent policy
learning. The critic receives detached learned event-state features and scalar
verifier returns; it is not part of the runtime checkpoint. On seed 19 with
three fresh transfer initializations, parent acquisition improved for some
fresh learners, but the retained model failed its stable retention gate and
the transferred parent was not qualified.

The mechanism is rejected as the default training path. It remains opt-in for
future variance-reduction experiments until its loss weighting and interaction
with the parent/retention phase transition are isolated.
