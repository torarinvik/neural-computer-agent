# Temporal behavioral-signature transfer rejection

This directory preserves the matched two-seed audit of behavioral artifact
signature v2. It is negative evidence and is not a promotion record.

Version 2 extends the previous two-step functional descriptor to eight fixed
four-step sequences. Each opaque external-compute artifact is run through the
same standardized learned event tensors and frozen controller intentions in
two modes: continuous execution and two reset segments. The policy receives a
fixed-width projection of the register, readout-intention, and decoder-logit
traces. It receives no raw parameter coordinates, family IDs, verifier labels,
correct actions, or reset metadata.

The real-file mechanistic screen passed: one learned file reached fresh
accuracy `1.0000`, and its v2 signature was finite, normalized, fixed-width,
and safely restored. The full matched audit then failed to transfer inherited
eviction knowledge:

- Seed 17: inherited `0/8`, fresh `8/8` transfer selections.
- Seed 18: inherited `0/8`, fresh `2/8` selections (`0.7500` overall fresh accuracy).

Both seeds mastered all six source files and the held-out n-back-2 file at
`1.0000`, retained the held-out file at `1.0000`, kept the controller and event
frontend frozen, passed candidate-order permutation, recorded no harmful
safety-gate probe, and replayed zero examples. The failure is therefore not
file acquisition, temporal probing, retention, or safety. It is a failure of
the inherited policy to learn a transferable utility model from source-family
outcomes.

The next high-value direction is not simply a larger signature. It is a
leave-one-family-out utility objective with explicit uncertainty and
calibration: separate “which artifact is useful” from “which artifact is safe
to evict,” train against held-out behavioral utility, and require a fresh
learner plus confidence-gated transfer. The verifier safety gate remains
necessary but is not sufficient.

Reports:

- `seed17.json`
- `seed18.json`

