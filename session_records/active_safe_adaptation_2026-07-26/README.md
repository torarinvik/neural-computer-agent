# Active verifier allocation — pre-registration

## Hypothesis

Fixed random contexts waste verifier bits whenever incumbent and challenger
choose the same operation. Preferentially verifying contexts where their
policies disagree should increase promotion evidence per scalar outcome while
fresh confirmation preserves safety.

## Mechanism

For each 60-outcome update, generate 240 unlabeled candidate contexts from the
ordinary sensory/controller pipeline. Rank them using only whether incumbent
and challenger latent action outputs disagree. Verify up to 60 disagreement
contexts; if fewer exist, fill the remainder with ordinary contexts.

The selector cannot see either action's outcome, the correct action, task
identity, or private audit metrics. Attempted action remains randomized at
propensity `0.5`. Proposal and confirmation use separate outcome blocks. During
confirmation, selection is based on the frozen proposal versus incumbent.

Accounting reports both 720 verifier bits and 2,880 unlabeled candidate
contexts. This is not “free data”; it explicitly trades cheap unlabelled
perception/compute for scarce verified outcomes.

## Hard-stream diagnostic

Replay failed seed 7973 with global-centered evidence, learning rate `0.003`,
mandatory confirmation, and pool multiplier four. It passes only if:

1. the mastered incumbent is never promoted or degraded;
2. the gap learner obtains a positive proposal within 480 bits;
3. a fresh block confirms it by 720 bits;
4. audited gap utility improves by at least `0.02`;
5. retention and persistence checks pass.

Only a full pass permits unchanged fresh seeds 7981 and 7982. Both fresh seeds
must pass before persistent skill integration is retried.

## Result

The active selector made the previously failed gap stream propose at 480 bits
with lower bound `+0.0024`, while the mastered incumbent remained untouched.
Fresh confirmation rejected the proposal (`lower95=−0.0827`), so nothing was
deployed. The challenger nevertheless improved private audited utility by
about `2.7` points.

Active allocation therefore improves proposal sensitivity but does not by
itself produce a reliably better challenger. The next fork changes the
learner's outcome objective while retaining active allocation and independent
confirmation.
