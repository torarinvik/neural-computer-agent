# Acquisition diagnostic: six interventions, one partial win (2026-08-08)

Plant acquisition reliability is the constraint gating every downstream
memory claim (F25): hard motor games acquire on one seed and not the
next. This record is the systematic sweep, all at matched budget (500
updates, batch 16, steps 32, egocentric roll), solo per game.

## Result

| intervention | forageA | intercept1 | collect1 | verdict |
| --- | ---: | ---: | ---: | --- |
| baseline (h=32, gamma .95) | 0.453 | 0.312 | 0.547 | reference |
| per-timestep baseline | 0.484 | 0.062 | — | no |
| + entropy bonus 0.01 | 0.406 | 0.188 | — | no |
| normalized advantage | 0.047 / 0.203 | 0.109 / 0.172 | 0.719 / 0.078 | no |
| hidden 64 | 0.031 | 0.078 | — | no |
| hidden 128 | 0.031 | 0.031 | — | no |
| **learned critic** | 0.469 / 0.281 | 0.453 / 0.141 | **0.812 / 0.859** | **partial** |
| critic + gamma 0.99 | 0.172 / 0.188 | 0.219 / 0.328 | — | no |

Two seeds shown as `69316 / 69317` where measured.

## Findings

1. **State-independent variance reduction does not work here.** Scalar
   baselines, per-timestep baselines, entropy bonuses and advantage
   normalisation are all neutral or harmful. Normalisation is the worst
   (forage 0.45 -> 0.05): dividing by the deviation amplifies noise into
   full-size gradients while the policy is still near-random, which is
   exactly when these games have no signal.
2. **Capacity makes it worse.** Quadrupling the controller collapses both
   motor games, so the constraint is not expressiveness — a larger
   recurrent policy under REINFORCE has a harder landscape to descend.
3. **A learned critic is the one thing that helps, and only where the
   failure is credit assignment.** collect (longest reward horizon)
   improves ~55% on both seeds. forage and intercept, whose failures are
   exploratory and timing-shaped, stay a seed lottery.
4. **Longer horizons do not compensate.** gamma 0.99 with the critic is
   worse on three of four measurements.

## Standing conclusion

Acquisition is not one problem. Credit assignment is real and now
partially solved (the critic ships, defaulting on where it helps).
What remains is exploration: games where the policy must find a rare
first success before any signal exists. The admissible levers there are
representational (F22/F28 moved these games more than any trainer
change) or algorithmic; scripted-expert bootstrapping is NOT admissible,
since it would inject the rules the verifier-private discipline exists
to withhold.
