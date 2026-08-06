# Twelve-addition extension-budget control — rejected

Status: rejected; doubling extension updates did not repair the twelve-step
retention boundary.

This control kept the generated length-six twelve-addition curriculum and all
promotion gates unchanged, but increased each new route-extension trainer
from 128 to 256 updates. Seed 69316 passed. Seed 69317 failed more strongly:
the final route fell to `0.75`, its capability remained unprotected, and the
fully protected-bank/recovery gates failed.

The result rejects “train every extension longer” as a reliable solution. It
also shows why lowering the retention threshold would be misleading. The next
fix must address route interference and confidence-aware admission under a
growing candidate bank, rather than only increasing per-extension optimizer
budget.

Evidence is in `report_seed69316.json` and `report_seed69317.json`.
