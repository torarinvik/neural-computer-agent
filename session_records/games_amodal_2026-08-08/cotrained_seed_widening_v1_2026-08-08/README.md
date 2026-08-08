# Co-trained self-addressing: 16-seed widening (post-probe scoring)

Corrected co-trained loop (`experiments/games_amodal/probes/cotrained.py`,
F53 post-probe scoring) at `--ignorance 2.0 --ignorance-every 1`,
16 seeds (69316-69331) run in parallel on a Vast.ai 192-core instance,
torch 2.12.0+cu130 (CPU). Plus the local torch runs of the same config
(`fix-base-*`) and the symmetric-plant variant (`fix-sym-*`, plus two
remote partials sym-69324/69330; the remaining 14 symmetric remote runs
were abandoned when the instance was closed -- the mechanism was already
failing and confirming a refutation at n=16 was not worth the rental).

Bar (pre-registered in probes/BARS.md): both twins mastered (>=0.9),
cross-feed <=0.1 both directions, sampled decoy within two batch quanta
of the post-probe no-agent floor (A 0.383, B 0.289), all on one seed.

Result: full bar 5/16; mastered-both 8/16; cross-feed inverts 10/16;
decoy collapses 11/16. Acquisition is the binding constraint; among the
8 seeds that master both twins, 7 invert and 5 clear everything.

Cross-platform note: the same seed does NOT reproduce across torch
2.12/Linux vs local macOS (local 69316 passed the full bar; remote 69316
failed cross-feed). Per-seed identities are platform-bound; only rates
transfer. Both platforms' runs are included here.
