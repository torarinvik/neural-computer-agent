# Replicated external-capability positive transfer (2026-08-06)

Status: replicated promoted replay-free transfer result.

One protected external file first learned:
`reverse -> adjacent_xor -> complement -> prefix_parity`. An inherited clone
of that file and a fresh external stack then learned the new target
`prefix_parity -> global_parity -> rotate -> complement` from fresh outcomes
only. A stable-prefix selector compared both learning curves; only the unique
winner was admitted beside the protected source file.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| source stable bits | 10,240 | 12,288 |
| source behavior | 0.9336 | 0.9531 |
| inherited target stable bits | **6,144** | **4,096** |
| inherited target behavior | 0.9961 | 1.0000 |
| fresh target stable bits | 10,240 | 10,240 |
| fresh target behavior | 0.9531 | 0.9570 |
| fresh-over-inherited ratio | **1.667x** | **2.500x** |
| source retention floor | 0.9336 | 0.9336 |
| reloaded source behavior | 0.9297 | 0.9609 |

Both runs selected the inherited candidate uniquely, grew the protected
capacity from one to two only after selection, preserved the source file,
rejected corrupted source snapshots, kept the controller digest unchanged,
and used zero replayed examples. The short rung correctly admitted nothing
when neither candidate had a stable prefix.

This is the first replicated evidence that an isolated external capability can
improve acquisition of a genuinely different future computation without
replaying old examples. It remains bounded: the transfer prior is an external
artifact, the candidate comparison is verifier-gated, and broad general
continual learning across arbitrary tasks is not yet established.
