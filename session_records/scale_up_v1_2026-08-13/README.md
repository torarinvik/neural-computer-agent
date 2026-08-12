# The scale-up (F220): 12x12, VALUES=12, constants only

3 seeds x 16 worlds, F216 protocol. Plant gates 1.0/0.9995/0.9987 at
the unchanged 40k budget. Sign -1 chosen on exactly avoid1/avoid2 all
seeds. signed - random +0.9069 +- 0.0669, t=+13.55, 48/48. Avoid at
92%/95% of the zero ceiling (was 88%/79% at 8x8). The 8x8 fingerprint
(intercept negative-but-improved, bank>oracle on avoid) reappears
unchanged. Not tested: slot-count changes, new mechanics.

Reproduce: python -m experiments.games_amodal.probes.scale_up --seed S
