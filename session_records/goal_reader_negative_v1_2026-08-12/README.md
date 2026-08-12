# The goal reader does not work (F217, negative)

Four designs (gr=dense reward, gr2=sparse+pure-held, gr3=512-transition
evidence, gr4=event-normalised features), 3 seeds each. The reader
never beat the constant mode-goal and never separated from the
reward-scrambled control. gr3 localises the boundary: pairs readable
from dynamics (5/6 on avoid worlds), sign unreadable because random
policies see ~1% reward events -- goal information is absent from the
evidence, not merely unexploited. The probe file is the gr4 version;
earlier variants are one-hunk diffs described in its comments.

## F218 -- the repair: read + verify reaches parity (gr5, gr6)

gr5 adds sign verification (2 rollouts): avoid worlds identical to the
search to 3 decimals, sign 6/6. gr6 adds the mode pair to the beam
(4 rollouts total): read_beam - searched = -0.0305 +- 0.0415, t=-0.74
at 1/90th the cost; beats mode t=+2.32; 21/21 over random. Both beam
components necessary: reader pair carries avoid (mode pair is absent
there), mode pair rescues collect1. goal_reader_final.py is the gr6
version.
