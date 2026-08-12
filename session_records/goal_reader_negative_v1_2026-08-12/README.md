# The goal reader does not work (F217, negative)

Four designs (gr=dense reward, gr2=sparse+pure-held, gr3=512-transition
evidence, gr4=event-normalised features), 3 seeds each. The reader
never beat the constant mode-goal and never separated from the
reward-scrambled control. gr3 localises the boundary: pairs readable
from dynamics (5/6 on avoid worlds), sign unreadable because random
policies see ~1% reward events -- goal information is absent from the
evidence, not merely unexploited. The probe file is the gr4 version;
earlier variants are one-hunk diffs described in its comments.
