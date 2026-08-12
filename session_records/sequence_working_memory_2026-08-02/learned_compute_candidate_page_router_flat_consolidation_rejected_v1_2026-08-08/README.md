# Flat multi-generation router consolidation — rejected

This audit tested replacing two independent generation routers with one flat
token-preserving router over all 18 append pages. The candidate was trained
only from newly generated scalar verifier outcomes, with no replayed examples;
the original source router and page-local screens remained frozen.

The narrow candidate (latent 32, hidden 64) reached generation accuracies
`0.50/0.50`; a wider candidate (latent 64, hidden 128) reached `0.50/0.5682`.
Permutation accuracy was `0.50` and `0.5341`, respectively. Both failed strict
per-target and per-page retention despite near-zero training loss and a valid
reward-shuffled null. The failure is therefore shared-page interference and
not simply insufficient hidden width.

The flat architecture is rejected. The next design is factorized
consolidation: one shared local-page router plus a small generation selector,
preserving the learned separation between generations while reducing the
number of full routers.
