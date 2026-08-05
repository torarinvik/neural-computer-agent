# Replicated online append-to-route growth

This is the end-to-end online memory test. The complement artifact is written
first into a deployed canonical bank and evaluated. A second
complement-reverse artifact is then appended later. The expanded bank is
reloaded and routed by a replaceable opaque outcome-trained router while the
controller remains frozen.

| Seed | Old skill before / after append | Complement selected / wrong / zero | Complement-reverse selected / wrong / zero | Route / shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 68001 | 76.6% / 76.6% | 75.9% / 29.1% / 29.5% | 63.6% / 28.4% / 28.4% | 100% / 50% |
| 68002 | 76.7% / 76.7% | 77.7% / 29.5% / 30.0% | 64.5% / 27.7% / 27.7% | 100% / 50% |

Both runs used the same frozen parent and independently acquired artifacts.
Appending the second row did not change the first artifact's behavior. Both
rows were selected, wrong addresses were discriminative, selected growth was
causal, the controller stayed bit-identical, and corruption was rejected.

This is a narrow replicated continuous-learning result: external memory can
grow after deployment and make a new frozen-controller capability reachable
without forgetting the old one. It does not yet establish online acquisition
of an artifact from raw experience, unbounded capacity, or general cognition.

Reports:

- `report.json` — seed 68001
- `../online_routed_growth_replication_2026-08-04/report.json` — seed 68002
