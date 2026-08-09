# Provisional external model candidate promotion

This fast audit tests the missing partial-evidence mechanism directly. A
novel transition stream is staged in a provisional model outside the
committed bank. The candidate retains its cumulative verified evidence window
and current updates train against that window; protected committed slots are
never replayed or modified. A held-out transition bundle and a caller-owned
retention probe must pass before the candidate is appended to the bank.

The controller is frozen, the old slot is digest-protected, and bank content
must remain unchanged before promotion. This is the intended copy-on-write
boundary for learning from partial streams without duplicate contexts or
catastrophic damage to committed memory. The first sparse-evidence audit was
correctly rejected; cumulative candidate evidence then passed both seeds with
held-out errors `0.129` and `0.143` at tolerance `0.2`. The promoted reports
are archived under
`session_records/sequence_working_memory_2026-08-02/external_provisional_candidate_promotion_promoted_2026-08-09/`.

The gain is bounded and explicitly replay-accounted: four unique target rows
were presented 600 times to the provisional candidate, while old committed
slot replay stayed zero. It is not yet replay-free general continual learning.

The router now also supports multiple isolated provisional candidates. Each
candidate has its own model, opaque context, evidence window, candidate-indexed
adaptation, and verifier-gated promotion. This prevents an alternating novel
stream from silently contaminating an earlier candidate; the focused
alternating-isolation regression covers staging, payload restore, promotion of
the second candidate, and byte stability of the first.

When the committed bank is full, `promote_staged_candidate` can now request a
larger bounded capacity in the same copy-on-write transaction. Capacity is
expanded only in the disposable candidate bank; held-out prediction and the
retention probe must pass before the live capacity and model slot are changed.

```text
.venv/bin/python experiments/external_provisional_candidate_promotion/train.py \
  --seed 70611 \
  --report-out /tmp/external-provisional-candidate.json
```
