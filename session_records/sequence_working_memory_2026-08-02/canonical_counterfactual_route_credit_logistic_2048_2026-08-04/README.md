# Counterfactual route credit — 2026-08-04

Primary seed `69411` passed the fresh-data route audit with paired
common-random counterfactual credit:

- route accuracy: `100%`;
- candidate permutation accuracy: `100%`;
- shuffled-outcome accuracy: `33.3%` three-way chance;
- selected execution: span 2 `100%`, span 3 `95.31%`, span 4 `88.67%`;
- route verifier bits: `131,072`;
- counterfactual pairs: `65,536`;
- replayed examples: `0`.

The mechanism is trainer-only and uses a bounded pairwise preference loss for
multi-row routing. It is a narrow credit-assignment result, not general
long-horizon credit or arbitrary program induction.
