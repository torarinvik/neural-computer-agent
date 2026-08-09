# Provisional external model candidate promotion

This fast audit tests the missing partial-evidence mechanism directly. A
novel transition stream is staged in a provisional model outside the
committed bank. Current evidence updates only that candidate. A held-out
transition bundle and a caller-owned retention probe must pass before the
candidate is appended to the bank.

The controller is frozen, the old slot is digest-protected, and bank content
must remain unchanged before promotion. This is the intended copy-on-write
boundary for learning from partial streams without duplicate contexts or
catastrophic damage to committed memory.

```text
.venv/bin/python experiments/external_provisional_candidate_promotion/train.py \
  --seed 70611 \
  --report-out /tmp/external-provisional-candidate.json
```
