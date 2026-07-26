# Persistent physical stream handoff — 2026-07-26

## Result

The controller now makes sequential replacement decisions against the same
small set of bounded, serialized physical memory banks. Later decisions derive
age, access, success, and failure features from the reloaded stores rather than
fresh synthetic history tensors.

## Accepted replicas

- Seed 7022, selected physical-online checkpoint 7012: 8 banks, 6 decisions,
  2.14 seconds.
- Seed 7023, independent physical-online checkpoint 7015: 8 banks, 6
  decisions, 1.92 seconds.
- Both crossed old-equal → reliability-dominant → old-return utility phases.
- Both passed exact save/reload, bounded-capacity, transition-accounting,
  physical/tensor parity, replacement, and causal-history-corruption gates.

## What is causally established

- Winning replacements persist into later decisions.
- Ordinary content-addressed reads and binary outcomes change the histories
  used by later decisions.
- Replaced rows correctly discard their old local statistics; every other
  history increment is accounted for exactly.
- Rolling access/outcome histories among physical rows changes at least one
  replacement action.
- Tensor evaluation remains only a shadow audit.

## Honest boundary

The controller weights were frozen in this probe. This demonstrates long-lived
physical state and causal use, not faster learning or successful online
adaptation over that state.

## Next smallest experiment

Run the three-candidate reward horse race while the same banks stay alive.
Compare utility-switch recovery, reward, verifier bits, and wall time against
the fresh-bank baseline. Start with an under-one-minute pilot and enlarge only
the candidate-estimation dimension if the physical reward gaps are noisy.
