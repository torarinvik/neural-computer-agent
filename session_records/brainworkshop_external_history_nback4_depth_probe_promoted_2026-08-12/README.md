# External-history n-back-4 depth probe (2026-08-12)

This pressure test checks whether the corrected bounded external-history
boundary can acquire a deeper temporal dependency. A fresh external file
receives five active records: the four preceding learned events plus the
current event. The complete event lifetime remains in append-only external
storage, and the shared controller and frontend remain frozen.

Seeds 17 and 18 both reached `1.0000` on every one of four fresh n-back-4
probe lifetimes after 192 attempted-outcome updates. The input boundary is
generic learned event tensors and scalar action outcomes; no rule name,
correct action, or privileged state enters the file.

This promotes bounded active depth four over append-only external history. It
does not establish multi-file n-back-4 retention, unrestricted learned query
depth, learned compression, arbitrary program induction, or general continual
learning.
