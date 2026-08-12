# Goal sign discovered per world (F216)

fixed-gl-*.json.gz: 3 seeds, 16 worlds, arms approach/avoid/signed
(forced) + free choice. Sign -1 chosen on exactly avoid1/avoid2, 3/3
seeds. Signed - random on pure avoid: +0.1484 +- 0.0248, t=+5.99, 6/6;
penalty removed 88%/79% of the attainable (ceiling is 0.000 -- avoid
worlds have no positive reward, so prediction 1 "goes positive" was
ill-posed). Bank beats oracle under the signed goal (-0.016 vs -0.099):
the flee-nearest proxy is still mismatched; goal over all hazards is
the open next step.

BUGGY-emptybank-*: kept because opposite goals returned byte-identical
behaviour -- build_bank dropped every row on worlds with absent slots
(third appearance of the F155/F192 row-vs-slot lesson). The tell is in
the instrument list now.

Reproduce: python -m experiments.games_amodal.probes.goal_language
--seed S. Single-threaded.
