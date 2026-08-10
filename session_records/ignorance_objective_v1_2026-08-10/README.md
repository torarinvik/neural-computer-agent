# F58's ignorance objective closes the twin collapse (F107)

F106 localised the defect: outcome model gave 0.9998 label agreement between an
entry and its INVERTED TWIN's; reader emitted entries 0.9855 cosine-similar for
opposite worlds. F58's remedy — penalise being accurate WITHOUT the entry —
applied as an entropy term pushing the entry-free prediction toward uniform.

## Model gate, checked FIRST (F106's standing rule)

| arm | twin agreement | food gap | entry cosine | outcome bal acc |
| --- | ---: | ---: | ---: | ---: |
| no ignorance | 0.9998 | 0.0000 | 0.9855 | 0.4312 |
| ignorance 0.5 | 0.5343 | 0.1473 | 0.7119 | 0.4305 |
| ignorance 2.0 | 0.6838 | 0.0863 | 0.3804 | 0.4321 |

Discrimination bought without costing prediction accuracy.

## Behaviour

| arm | held-out | twin entry | entry effect |
| --- | ---: | ---: | ---: |
| no ignorance | -0.0466 | -0.0471 | +0.0005 |
| ignorance 0.5 | -0.0217 | -0.0716 | +0.0499 |
| best invariant policy (ref) | -0.0318 | | |
| oracle (ref) | +0.1954 | | |

Both seeds: entry effect +0.0543, +0.0455. Beats the best context-free policy
for the first time in F100-F107. The wrong rule now HURTS (-0.0716 vs a
withheld -0.0480) — F99's signature on a multi-step task.

**22.0% of the measured headroom captured by reading context.**

## Scope

22% is not 100%. And the response is NON-MONOTONE: ignorance 2.0 gives more
differentiated entries (cosine 0.3804) but worse behaviour (+0.0230). More
pressure is not simply better; the curve is being swept.
