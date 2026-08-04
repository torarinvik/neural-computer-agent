# v60 stable content-address boundary

v60 tests the direct architecture correction identified by the v59 audit:
memory keys are now optionally formed from the learned event payload alone,
excluding age, duration, timestamp presence, and confidence transport
features. This makes the address invariant when the same event is written and
later recalled at a different event age.

The three-seed run preserves the parent and causal retention gates, and
persistent reload/checksum/recovery pass. Unseen-token recall improves over
the identity diagnostic on average but remains variable (`0.789`, `0.738`,
`0.844`). Transfer qualifies for two of three seeds, so the population
transfer claim remains rejected.

The payload-only address is promoted as the v24 interface behavior because it
fixes a write/read contract violation. It is not promoted as a learned
retention capability gain, and no v60 checkpoint is curated.
