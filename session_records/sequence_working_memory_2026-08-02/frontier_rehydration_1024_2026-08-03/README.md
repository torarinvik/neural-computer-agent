# Protected frontier rehydration (pre-registration, 2026-08-03)

The small continuation and routing screens did not improve the accepted
frontier. This rung tests the remaining data-limited hypothesis using the
same successful missing-evidence recipe at a larger fresh-target budget:

- parent: `artifacts/checkpoints/span11_missing_evidence_rehearsal_seed996047.pt`;
- 512 target span-11 mixed lifetimes;
- 512 span-10, 512 span-9, and 512 blank span-11 protected lifetimes;
- 32 epochs, batch size 512, learning rate 0.0005;
- binary complement and outcome-conditioned critic losses, gate/logit
  protection 0.1;
- no new read, routing window, difficulty weighting, or semantic branch.

The target is a positive child-over-parent gain with the same causal, old
retention, blank, and reset gates. The audit ladder is 1,024 lifetimes first,
then 4,096 only if the high-power acquisition interval is positive.
