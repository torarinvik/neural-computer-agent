# Rejected three-procedure independence gate

This stress test added a third 512-step `complement_rotate` artifact to the
promoted `complement` and `complement_reverse` bank. The router itself was
strong: 100% normal routing, 1/3 reward-shuffled routing, 100% row
permutation, 1/3 cosine baseline, and all three rows selected.

The full independence gate failed because the complement and complement-
rotate artifacts transferred to each other's procedures almost perfectly:

- complement selected 77.2%, wrong row 2 76.1%, zeroed 28.9%;
- complement-reverse selected 63.3%, both wrong rows about 28.3%;
- complement-rotate selected 64.8%, wrong row 0 64.8%, zeroed 30.8%.

All selected artifacts were causal, but not every alternate artifact was
behaviorally discriminative. This is retained as a negative control showing
shared learned subroutines, not promoted as a three-independent-procedure
result.

Report: `report.json`.
