# Model + value + search on the games: the entry does nothing (F101)

F100 located its failure in greedy one-step action selection. This builds F67's
missing half: transition model (cell, action) -> next cell, value model
(screen, cell, entry) -> outcome of standing there, and BFS over both,
recomputed every step.

| arm | reward | floor | lift | beats floor |
| --- | ---: | ---: | ---: | ---: |
| trained variants | -0.0385 | -0.0402 | +0.0017 | 7.0/12 |
| held-out variants | -0.0383 | -0.0400 | +0.0016 | 6.5/12 |
| entry WITHHELD | -0.0380 | -0.0400 | +0.0020 | 6.5/12 |
| STRANGER entry | -0.0377 | -0.0400 | +0.0023 | 6.5/12 |
| random plant | -0.0444 | -0.0436 | -0.0008 | 5/12 |

Search doubled the lift over greedy (+0.0016 vs +0.0008) but both are
negligible. The decisive column is the nulls: withholding the entry scores the
same and a STRANGER'S entry scores slightly better. On `dual` the same nulls
were brutal (stranger entry drove reward to -0.100, F99). Here the entry is
decoration.

## Diagnosis: the state, not the derivation

Avatar cell + one screen frame is Markov-insufficient for these games.
`intercept` has objects falling, `avoid` has hazards moving; a single frame has
no velocity or phase. "Cell c is safe now and lethal in two steps" is not
expressible, so the model predicts something its inputs do not determine and no
search or entry can rescue it.

Same boundary as F92 one level up: F92 found failure on dynamics that are
functions of the state's IDENTITY; this is failure on dynamics that are
functions of state HISTORY.

## Next

A factored MULTI-OBJECT state — avatar, each faller and hazard, and enough
frames to expose motion. That is exactly the slot interface of F71-F98, which
has never been fed game objects: `schema_families.py` handles six slots of eight
values and a composigrid frame with an avatar and two hazards is that shape.

Recorded rather than attempted: two formulation failures in a row is where this
project's rules say to stop iterating and state the finding.
