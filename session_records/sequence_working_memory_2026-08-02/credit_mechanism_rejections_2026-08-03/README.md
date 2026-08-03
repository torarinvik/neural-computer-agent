# Credit-mechanism screen rejections (2026-08-03)

These follow-up arms were run after the target-diversity screens and before
the successful population race. Every arm retained the missing-evidence and
old-span controls; none was promoted.

| Arm | Target audit | Child-over-parent | Causal gap | Decision |
| --- | ---: | ---: | ---: | --- |
| event-age successor | 256 | +0.04 pp | +0.04 pp | reject; CI crossed zero |
| event-age + scalar difficulty | 256 | −0.32 pp | −0.32 pp | reject |
| 512-target frontier rehydration | 1,024 | +0.23 pp | +2.91 pp | reject; CI crossed zero |
| joint action adapter | 256 | −9.45 pp | +0.11 pp | reject; retention/control failure |
| workspace/usage/age successor | 256 | −0.11 pp | −0.11 pp | reject |
| detached critic-to-policy | 256 | −1.46 pp | +4.08 pp | reject; acquisition failed |

The positive causal gaps in the critic and event-age arms again demonstrate
why newest-slot use is not enough: the child must beat its parent on held-out
target behavior. The action-adapter arm is especially important as a negative
control: unfreezing the projection caused roughly 10–11 point old-span loss
and pushed blank behavior away from chance.

No checkpoint from this directory is curated. The population race that
followed is recorded separately in
`missing_evidence_population_2026-08-03/`.
