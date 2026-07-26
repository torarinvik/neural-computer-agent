# Physical online adaptation handoff — 2026-07-26

## Result

The two-dimensional memory-utility controller now adapts from verified outcomes
measured through bounded serialized physical disk memories. Tensor evaluation
is a shadow parity audit only.

## Accepted runs

- Seed 7012: 91.06%, 82.13%, 88.23%, 82.91% across the four phases;
  136.33 seconds.
- Seed 7015: 85.74%, 77.25%, 86.72%, 82.67%; 136.69 seconds.
- Both used 48 updates, 196,608 unique verifier bits, zero replay, and
  persisted 6,144 complete histories.
- Both retained binary mapping and four-rule behavior; only
  `memory_replacement_extra_gate.weight` changed.
- Every selected physical candidate was equivalent to the tensor-shadow
  optimum within `1e-6`.

## Controls and rejected pilots

- Seed 7001 passed the 32-bank physical/tensor parity preflight exactly.
- Seeds 7010 and 7011 rejected 32-bank candidate estimates as too noisy.
- Seed 7013 shuffled the physical candidate rewards. It failed all four
  adaptation phases, learned the reliability coefficient in the wrong
  direction, and saved no checkpoint.
- An exact-index parity requirement was replaced by a pre-registered
  tie-aware equivalence check after one replica exposed a `5.96e-8`
  floating-point separation between otherwise tied candidates.

## Curated artifacts

- `artifacts/checkpoints/unified_memory_physical_online_seed7012.pt`
- `artifacts/checkpoints/unified_memory_physical_online_seed7015.pt`
- `experiments/unified_cognitive_controller/reports/physical_online_parity_seed7001_banks32.json`
- `experiments/unified_cognitive_controller/reports/physical_online_adaptation_seed7012.json`
- `experiments/unified_cognitive_controller/reports/physical_online_adaptation_seed7015.json`
- `experiments/unified_cognitive_controller/reports/physical_online_adaptation_reward_shuffled_seed7013.json`

## Honest frontier

Each update currently starts from fresh bounded banks. The next smallest
experiment keeps the same few banks alive across several updates and utility
switches, accumulating reads, verified outcomes, and replacements. It must
retain bounded size, exact persistence, causal corruption effects,
tensor-shadow parity, switch recovery, and old skills before any consolidation
or learned-statistic work begins.
