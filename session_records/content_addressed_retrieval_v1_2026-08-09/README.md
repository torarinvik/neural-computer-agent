# Content-addressed retrieval: constant plant cost (F86)

F85's measured failure: a linear scan costs N plant forward passes, so at N=64
recognising a task was dearer than minting one. Content addressing stores an
ADDRESS (key of the entry as first read) beside the CONTENT (tuned entry), and
matches a fresh read by cosine — one encoder pass, no plant passes.

| N | scan acc | scan passes | key acc | key passes | key+verify | verify passes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000 | 8 | 1.000 | 0 | 1.000 | 4 |
| 16 | 1.000 | 16 | 1.000 | 0 | 1.000 | 4 |
| 32 | 1.000 | 32 | 1.000 | 0 | 1.000 | 4 |
| 64 | 0.969 | 64 | 1.000 | 0 | 1.000 | 4 |

Both seeds 1.000 for keys at N=64, where the linear scan had slipped to 0.969.

## Cost problem solved

Retrieval is 4 plant passes (constant in N) vs minting's 2.7-7.0 update steps
(forward AND backward each). Recognising is now decisively cheaper than
relearning, and stays so as the bank grows.

## Keys cannot reject strangers — the verify step is not optional

| N | key in-bank | key stranger | gap | consequence gap |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.992 | 0.667 | 0.325 | 0.571 |
| 64 | 0.990 | 0.862 | 0.128 | 0.358 |

A never-seen family matches its nearest stored key at 0.862. Key-only reuse
above a threshold would reuse constantly — a bank that stops minting and starts
pretending. So:

- keys ADDRESS: 0 plant passes, perfect shortlist, no "none of these";
- consequence VERIFIES: 4 plant passes, constant in N, supplies "none of these".

This is ARCHITECTURE.md §2.3's cued-vs-probe split arriving from measurement,
and the first result showing the two routes are complementary rather than
competing.

## Gate status: all three clauses pass

(a) 64/64 mastered. (b) drift exactly 0.0. (c) acquisition cost flat, retrieval
1.000 at N=64, retrieval cost constant. (c) is genuinely falsifiable — F85
showed it failing on cost, F86 fixed the mechanism.

Watched, not extrapolated: key gap shrinks ~0.066/doubling, consequence gap
~0.071. Unmeasured beyond N=64; re-measure at 128 and 256.
