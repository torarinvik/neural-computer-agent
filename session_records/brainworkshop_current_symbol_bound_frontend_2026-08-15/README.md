# Bound current-symbol acquire on a curated frontend (2026-08-15)

Status: **replicated, not admitted**.

Unused seeds 119017, 120017, and 121017 each acquired a prototype-match
file against the same frozen rendered frontend
`rendered_frontend_seed1001.pt` (`1ce405a0…`). `AgentBrain.bank` was not
written. Slot 0 stayed `90e20193…`. This population does not reuse
116017–118017 or the Dual lease 113017–115017.

## Results

Shared frontend digest `1ce405a0…`. Controller digest `59c9ef2b…`.

| Seed | Train | Frozen hold | Zeros | Reverse | Shuffle | Delay slot 0 | Cross encoder |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 119017 | 0.958 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.542 | 0.792 | 0.500 |
| 120017 | 0.958 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.521 | 0.667 | 0.500 |
| 121017 | 0.938 / 48 | 1.000 / 48 | 0.500 | 0.000 | 0.521 | 0.604 | 0.500 |

## Same-frontend transfer (post-campaign)

The prototype acquired on 119017 then scored `1.000` frozen on 119018,
120017, 120018, 121017, and 121018 with zero program updates. A
different encoder stays at chance. That is a reusable bound template on
this frontend, still not a curated bank slot.

## Limits

- not a curated AgentBrain slot;
- 48-bit prefixes, not a ten-minute unused lease;
- not open program induction;
- not a Dual 2-back transfer ratio;
- not desktop Dual.
