# Goal-factored cued twins: 3/6 full bar, 4/6 both twins mastered

Rung A of `docs/GOAL_FACTORED_DESIGN.md`. Plant = goal-following
executor trained verifier-free on self-checkable micro goals, then
FROZEN. Bank = two destination fragments. Cue = the game's name
rendered in the world, read by a 2->2 cue-reader. Per-game gradients
touch only fragments + cue-reader.

`gfi-*` is the final configuration (phase-1 ignorance objective +
annealed cue entropy); `gfd-*` is the previous one, kept for the
matched comparison that motivated both mechanisms.

Floors (F52, measured, no agent): choiceA 0.344, choiceB 0.320.

    seed  draws  mastery A/B   cross A/B    decoy A/B    swap A/B
    69316   1    1.00 / 1.00   0.05 / 0.00  0.29 / 0.49  0.00 / 0.00
    69317   1    1.00 / 1.00   0.00 / 0.00  0.31 / 0.33  0.00 / 0.00  FULL
    69318   1    0.82 / 0.77   0.06 / 0.00  0.18 / 0.43  0.01 / 0.00
    69319   1    0.28 / 0.48   0.20 / 0.01  0.30 / 0.35  0.13 / 0.18
    69320   1    1.00 / 1.00   0.00 / 0.00  0.31 / 0.30  0.00 / 0.00  FULL
    69321   1    1.00 / 1.00   0.02 / 0.00  0.24 / 0.25  0.00 / 0.00  FULL

FULL BAR 3/6; both twins mastered 4/6; cross-feed inverts below floor
6/6; label-swap collapses below floor 5/6 (69319's 0.13/0.18 is below
floor too, but that run failed mastery so it is not evidence).

Phase-1 draws fell to 1 on every seed once the ignorance objective was
added -- an unpredicted side effect worth noting: teaching the plant to
be uninformative off-vocabulary appears to REGULARISE the conditioned
learning, removing the basin lottery that previously cost 2-6 draws.

Remaining failures: decoy on ONE twin at 0.43-0.49 (69316, 69318),
i.e. the residue of the default-response problem; and phase-2
acquisition on 69318/69319.

Comparison, same architecture question, monolithic co-trained line:
F55 measured 5/16 on its full bar with acquisition the binding
constraint. This rung is 3/6 with all-gates-passing runs having
mastery 1.00/1.00 exactly.
