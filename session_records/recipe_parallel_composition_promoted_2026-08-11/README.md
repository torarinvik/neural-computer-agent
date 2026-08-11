# Generic parallel recipe composition — promoted narrow rung

This two-seed in-repository audit tests the expressibility boundary suggested
by the exported games session. The export's learned-interpreter numbers remain
**SINGLE-SOURCE, UNREPLICATED**; this is an independent random-program audit.

The interpreter is trained only on random programs over six abstract slots and
eight values. No task family, semantic label, modality, or privileged answer
enters the learner. The baseline grammar contains `NOOP`, `INC`, `DEC`,
`CINC`, `CDEC`, `COPY`, and `SWAP`. The extension adds only the generic
`PARALLEL(left, right)` combinator, which applies disjoint local effects to the
same pre-step state and commits them atomically.

At the stable-prefix threshold `0.9`, both extended arms reach old-basis
length-two and double-length execution by update `1,500`. The paired
increment target—the two-valued pair-flip specialization—is also learned by
both extended arms. The baseline reaches the old-basis threshold at update
`2,000` on seed `70422` and never reaches it within `2,500` updates on seed
`70421`.

| seed | grammar | stable old length-2 | stable old length-4 | stable paired target | wall time |
| ---: | --- | ---: | ---: | ---: | ---: |
| 70421 | atomic | not reached | not reached | not applicable | 67.38 s |
| 70421 | + parallel | 1,500 | 1,500 | 1,000 | 81.42 s |
| 70422 | atomic | 2,000 | 2,000 | not applicable | 81.87 s |
| 70422 | + parallel | 1,500 | 1,500 | 500 | 81.18 s |

The extension therefore promotes a bounded expressibility and random-program
execution rung, not general continual learning, arbitrary Turing-complete
acquisition, or a claim that every richer basis is cheaper. The measured
parallel-arm wall-time cost averaged `8.9%` above the atomic arm. Exhaustive
symbolic candidates grew from `118` to `5,518` (`46.8x`), so a learned proposal
distribution will be necessary at scale.

## Confound and next control

The extended arm trains on a richer random-program distribution, so its
improved old-basis curve may reflect useful compositional training pressure as
well as the new operator's representational capacity. The next control should
hold the training program distribution fixed while exposing the new operator
only at evaluation, then separately measure the benefit of deliberate
composition practice. This is a follow-up, not a reason to retract the
expressibility result.

Reports and accounting are in this directory.
