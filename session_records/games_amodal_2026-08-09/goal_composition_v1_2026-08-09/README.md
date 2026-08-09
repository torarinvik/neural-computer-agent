# Zero-shot composition by slot assembly: it works

Rung B of `docs/GOAL_FACTORED_DESIGN.md`. Frozen goal-following plant;
a fragment is two slots (cue 0, cue 1) naming which side to want.
Held-out pairings are ASSEMBLED from slots trained in other games --
no gradient step, no new learning.

Per-holdout, the two NON-degenerate cases (see caveat):

    seed    holdout   assembled   scrambled   trained(mean)   floor
    69316   c02        0.569       0.271       0.702          0.333
    69316   c20        0.685       0.253
    69318   c02        0.641       0.233       0.698          0.333
    69318   c20        0.706       0.207

Assembled reaches 85% (69316) and 111% (69318) of the trained pairings'
performance on the floor-to-ceiling scale, with zero learning. The
scrambled control -- same two donor slots, plugged into the wrong cues
-- sits at or below the no-agent floor. So the assembly is doing the
work, not the donors merely being present.

CAVEAT, found by inspection and recorded rather than buried: the third
held-out pairing c11 is (cue0->side1, cue1->side1). Swapping its slots
is a NO-OP, so its "scrambled" control is degenerate and its numbers
(assembled 0.686/0.866 vs scrambled 0.698/0.873) are not evidence either
way. Only c02 and c20 carry the claim. compose_suite's holdout set was
not designed with a swap control in mind.

Seed 69317 is excluded: phase 1 never converged (competence 0.014 on all
three sides after 6 draws), so every downstream number sits exactly at
floor. It is reported as an executor failure, not a composition failure.

Standing limitation: arity-3 executor training is unreliable -- draws
3, 6, 5 across the three seeds, and side competence is uneven even when
it converges (0.29-0.88). Composition is demonstrated conditional on a
competent executor; making the executor reliable is the open work.
