# Three-instruction external-register composition: rejected diagnostic

Date: 2026-08-07
Status: rejected; diagnostic only
Schema: `neural-computer.external-register-three-instruction-report.v1`

This pressure test extended the promoted two-instruction rendered-event
result to a verifier-private reverse -> complement -> rotate program. A
frozen parent controller emitted learned events; three opaque instruction
vectors were acquired sequentially with no replay of earlier examples; then a
fresh decoder learned the triple composition while the parent and register
interpreter were frozen. The reversed-order program was evaluated separately
to test order sensitivity.

The first short rung was undertrained and was retained as a curriculum
diagnostic. The second rung passed primitive acquisition and retention:

| Measure | Short rung | Acquisition rung |
| --- | ---: | ---: |
| Reverse after third instruction | 0.9219 | 0.9961 |
| Complement after third instruction | 0.7188 | 0.9648 |
| Rotate | 0.7578 | 0.9180 |
| Frozen triple composition | 0.6172 | 0.6758 |
| Reversed-order composition | 0.5469 | 0.6836 |
| Fresh triple composition | 0.9063 | 1.0000 |
| Reward-shuffled composition | 0.5938 | 0.4531 |
| Missing-evidence composition | 0.5000 | 0.5000 |

The acquisition rung used `114,688` unique verifier bits, `28,672` unique
logical lifetimes, `1,920` optimizer updates, and zero replayed examples.
Frozen triple composition never reached the `0.8` stable-prefix mastery gate;
the matched fresh learner reached it at `16,384` bits. Exact reload,
checksum-corruption rejection, frozen-parent equality, and the causal controls
passed, so this is not evidence of an invalid persistence or reward path.

No capability was promoted and no checkpoint was curated. The result isolates
the next bottleneck: three acquired primitive vectors remain individually
usable after sequential learning, but the frozen serial interpreter plus a
fresh decoder does not yet expose a stable, learnable depth-three composition
state. The next implementation experiment should separate event ingestion
from program execution or provide an explicit execution/read boundary, then
re-run the two-instruction promotion regression before advancing depth.
