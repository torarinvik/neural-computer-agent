# Target-diversity and successor-slot screen rejections (2026-08-03)

Three matched screen axes failed before promotion from the accepted
missing-evidence frontier:

| Arm | Result | Reason for rejection |
| --- | --- | --- |
| suffix query window `[4, 11)` | −0.89 pp acquisition; causal gap +3.41 pp | no positive child-over-parent gain |
| matched full-window baseline | −0.32 pp acquisition; causal gap +3.98 pp | no positive child-over-parent gain |
| target position augmentation | −0.78 pp acquisition | no positive child-over-parent gain |
| appended zero-output slot | 0.00 pp acquisition and causal gain | newest slot did not contribute |

All four corrected 256-lifetime audits passed blank/reset chance controls and
old-span retention where applicable, but none passed the acquisition gate.
The positive zeroed-slot gaps in the continuation arms are not sufficient:
the current parent must also improve on held-out target behavior. No screen
checkpoint was curated.

Receipts for the training and corrected audits are stored in this directory.
