# Page-local source-budget split rejected (2026-08-07)

This matched control gives the normalized source page and raw prior 512
updates each, keeping the six raw append pages and all verifier controls
unchanged. Both seeds acquire every unseen candidate, but seed `69317` loses
strict known-source mastery at `0.8854`, with a minimum per-target score of
`0.4286` and three target holes.

The result rejects splitting the available source budget evenly. The
page-local representation blueprint remains useful, but the normalized source
page currently needs the full 1,024-update rung for replicated retention.
