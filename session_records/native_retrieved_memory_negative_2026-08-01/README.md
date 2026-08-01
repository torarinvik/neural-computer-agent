# Direct native `retrieved_memory` negative control — 2026-08-01

## Question

Can a small translator make the frozen controller use its existing
`retrieved_memory` vector input directly?

## Result

The translator was trained from 256 support contexts and 200 updates. On 2,048
held-out contexts it reached only 56.59% with outcome-only learning, 58.15%
after reversal, and 15.04% paired prediction flips. Exact disk retrieval and
the controller digest were intact, so this was not a storage or weight-update
failure.

A disposable capacity probe trained the same adapter with the private query
answer labels. It remained at 56.59% and 15.28% flips. Thus the native hidden
read interface is not merely hard to optimize from sparse outcomes; this
checkpoint does not expose enough useful action capacity through that vector
alone.

The reports are `outcome_only_seed29101.json` and
`private_label_capacity_seed29101.json`. The failed path is retained as the
boundary-localization control for the successful intention-bus reader.
