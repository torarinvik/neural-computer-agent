# Transfer matrix (F156)

Probe 254. 13 families, every source x every target, 2 seeds. Plant
trained on one source, FROZEN, only a bank entry fitted per target.

Donor strength (advantage over an untrained plant, given to OTHER
families):
    chaos +0.4883 | walled +0.4335 | dial +0.4215 | grid +0.4201
    perm +0.2680 | line +0.0896 | toggle +0.0874
    scrambled (CONTROL) -0.1741   <- actively harmful

The negative control validates the measurement: schema-destroyed
training hurts, so positive rows are transferable structure and not
warm start.

CAVEAT: donor strength correlates -0.452 with source slot count, so
part of the ranking is structural overlap with the target pool. Not
yet a curriculum.

Open question: chaos (no rule) is the best donor while scrambled (also
no rule) is the worst.
