# Library reuse is a null as implemented (F157)

Probe 255. Growth vs frozen library, paired per family, 2 seeds.

    paired mean cost ratio (grow/frozen) 0.944
    cheaper with growth in 7/11 comparable cases
    spread 0.14 to 2.33 — grid gives 0.26 on one seed and 2.33 on the
    other, which is noise, not mechanism

Cause, predicted before the run: fragments are appended whole and
sampled UNIFORMLY, so the library grew 210 -> 242 and a useful
fragment went from 1-in-210 to 1-in-242. Adding good fragments to a
uniform pool makes each one rarer. This is a null on the
implementation, not on reuse.

Also: search cost is dominated by family, not position (dial ~400,
toggle saturates 24000), and toggle is the family F155 flagged as
least expressible in the basis.
