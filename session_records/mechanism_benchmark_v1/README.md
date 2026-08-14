# F230: mechanism benchmark, first witnesses (2026-08-13)

mb/  mechanism_baseline.py, 6 seeds: frozen F226 stack on the DEV
     mechanism set. delayed3 PASS (+1.25 over random); resource1 HARD
     FAIL (+0.11 vs ~+1.5 reference) -> sequencing witness; deceptive1
     bait tax -0.29 (t=-1.98); collect2 anchor exact.

SEALED mechanisms (blink, oneway, lever): implemented and unit-tested
in experiments/games_amodal/game_family.py, never probed. The seal
holds until an explicit unseal decision recorded in the design doc.
