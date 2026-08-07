# Bank-26 six-stage pressure test rejected (2026-08-07)

This pressure test extends the promoted bank-20/five-stage configuration to
26 candidates: 14 source candidates plus 12 outcome-unseen candidates across
six isolated append stages. The source budget is 1024 updates and each stage
receives 32 fresh calibration updates with a full append prior.

All twelve unseen candidates are acquired at `1.0000/1.0000`, but the source
screen is not mastered: known routing is `0.9271/0.7500` and strict per-target
known mastery fails on both seeds. The hard seed also fails the
reward-shuffled null. This is rejected as a promotion, while localizing the
26-candidate boundary to source-screen capacity/interference rather than
append acquisition.
