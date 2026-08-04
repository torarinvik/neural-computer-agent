# Sequential contradiction reversal boundary

The sequential contradiction verifier was audited after its documented
default configuration exposed a harness error: `sample_random_reversal` had
required six blocks while the default sequence contained four. The minimum
prefix/suffix was corrected to one block and a regression test now covers the
default configuration.

The current v17 controller uses a neutral source-credit output bias,
normalized source-key attribution, and a mean update over present event
tokens. With frozen independent frontends, sequence length 8, two-tick
blocks, seed 17, 512 updates, and batch size 128, it reached `0.7002`
Markov-schedule reward overall and `0.5635` after the first hidden role
transition. Fixed reversal reached `0.8037`/`0.8099` pre/post-transition;
stream-order shuffling reached `0.6877` overall. No-feedback,
feedback-shuffled, action-shuffled, and intention interventions stayed near
chance. The 512-step run therefore passes the fixed-reversal gate but not the
Markov post-transition or stream-order gates.

A lower learning rate (`5e-4`) at 1024 updates produced one passing seed
(seed 17: Markov `0.7644`/`0.6223`, reversal post-transition `0.7702`,
stream-order `0.7478`), but seeds 18 and 19 failed different gates. The
three-seed population is therefore not promoted; this identifies optimization
stability and generalization across hidden schedules as the next bottleneck.

This is a genuine capability boundary after the transport/training harness
fix, not a promoted result. The historical fixed-role source-trust result
remains recorded separately under
`session_records/calibration_conflict_amodal_2026-08-03/`.
