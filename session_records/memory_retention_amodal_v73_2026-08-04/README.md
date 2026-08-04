# v73 matched token-block transfer qualification

v73 combines two implementation fixes with the v25 runtime: transfer arms
inherit the declared training configuration, including the retention-policy
reset, and randomized opaque token pairs are reused for four bounded episodes
before a new pair is drawn. The latter keeps the policy-gradient target
stationary briefly without exposing a task label or adding a controller branch.

The 1,024/1,024 three-seed rung passes the narrow retention gate on seeds 17,
18, and 19. All three retained models pass parent retention, causal clear and
corruption controls, target-first/target-last symmetry, four-pair unseen-token
recall, and persistent reload/checksum recovery. The minimum unseen-pair recall
is `0.727`; the mean across seeds is `0.912`.

The matched fresh-transfer population also qualifies on every seed. Fresh over
transferred stable-bit ratios are `2.103x`, `1.538x`, and `2.000x` (mean
`1.880x`; minimum `1.538x`). This is a promoted narrow outcome-only transfer
result, not a claim of broad natural-modality or general episodic memory.

Reports are `seed_17.json`, `seed_18.json`, and `seed_19.json`. The trainer
default now uses the four-episode token block when token randomization is
enabled; fixed-token historical runs are unchanged.
