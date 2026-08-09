# Provisional candidate promotion — rejected audit

This audit exercised the new copy-on-write boundary. A novel stream staged a
candidate after two rows, updated only the provisional model from current
evidence, and left the committed bank unchanged until promotion was attempted.

The isolation gates passed on both seeds: the controller stayed frozen, the
source slot remained byte-stable, and the bank content digest did not change
while the candidate learned. Promotion was correctly rejected because the
candidate failed the held-out prediction gate:

| metric | seed 70611 | seed 70612 |
| --- | ---: | ---: |
| provisional updates | 200 | 200 |
| held-out error | 3.203 | 0.665 |
| prediction tolerance | 0.2 | 0.2 |
| bank write before promotion | no | no |
| promotion | rejected | rejected |

Verdict: reject capability promotion. The staging API is safe, but four
current transition rows are not enough for this model to generalize to the
held-out rows. The next mechanism is evidence-aware candidate acquisition:
decide when a candidate has enough diverse coverage to promote, and use
structured/current evidence to improve it without replaying protected slots.

Reports are protected by `SHA256SUMS`.
