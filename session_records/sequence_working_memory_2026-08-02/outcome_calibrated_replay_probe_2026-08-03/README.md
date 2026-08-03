# Outcome-calibrated replay probe — 2026-08-03

This is a retrospective calibration diagnostic following the rejected
parent-to-child policy-KL stopper. The proposed controller compares paired
scalar verifier outcomes from a frozen parent and candidate on a fresh stream
and on each protected stream. It stops only when the fresh acquisition lower
bound clears the minimum gain and every protected-stream lower bound remains
inside the retention tolerance.

Every diagnostic logical lifetime is charged to the replay budget. The probe
does not receive correct actions, task labels, logits, or causal ablations;
those remain private to the later verifier audit. A budget-exhausted result is
reported as inconclusive rather than successful.

The first run is deliberately small and retrospective on the archived 512
replay lineage (`span11_prior_adaptive_seed996033.pt`). It is a calibration
smoke, not a promotion claim and not evidence that the 996033 checkpoint was
selected autonomously.

## Calibration result

The 64-lifetime smoke charged 512 replay lifetimes plus 192 diagnostic
lifetimes to a 1,024-lifetime budget. Acquisition was estimated at +1.28 pp,
but the protected span-9 estimate was −3.82 pp with its 95% interval below
zero, so the controller continued.

A single 256-lifetime diagnostic escalation charged 512 replay plus 768
diagnostic lifetimes to a 2,048-lifetime budget. Acquisition was +2.34 pp
(95% lower bound +1.16 pp), while protected span 9 was −1.74 pp (lower bound
−3.03 pp) and span 10 was −0.82 pp (lower bound −2.11 pp). The controller
continued because retention was not yet safely established.

This is a useful negative for the controller design: scalar outcomes provide a
causal acquisition signal, but this paired per-lifetime estimator is too noisy
to authorize an autonomous stop at this budget. Keep the verifier-side causal
and retention audits private, and do not promote this policy or scale it until
an independent calibration shows that it preserves the two-point retention
gate.
