# Persistent physical reward adaptation — 2026-07-26

## Result

The controller now adapts its two-weight generic memory-utility residual from
verified physical outcomes while the same disk-backed banks remain alive.

Selected run:

- seed: 7032
- device: CPU
- persistent banks: 16
- capacity: 6
- decisions: 9 (three per utility phase)
- duration: 24.28 seconds
- checkpoint:
  `artifacts/checkpoints/unified_memory_persistent_physical_seed7032.pt`
- report:
  `experiments/unified_cognitive_controller/reports/persistent_physical_adaptation_seed7032_banks16.json`

## Main evidence

In the reliability-dominant phase:

- learned target-row selection: 50.0%
- frozen target-row selection on identical states: 14.58%
- learned verified future reward: 94.79%
- frozen verified future reward: 91.67%

When old-equal utility returned:

- learned verified future reward: 93.75%
- frozen verified future reward: 93.40%

All physical banks stayed at capacity, all state transitions and save/reloads
were exact, physical reward matched the tensor parity shadow within `1e-6`,
binary and four-rule retention passed, and only
`memory_replacement_extra_gate.weight` changed.

## Adversarial control

Seed 7033 shuffled the three physical candidate rewards before selecting the
winner. It failed:

- reliability target-row selection: 6.25%
- frozen target-row selection: 20.83%
- learned verified reward: 88.89%
- frozen verified reward: 91.32%
- old-return recovery also failed
- no checkpoint was saved

This rules out persistence and random perturbation as sufficient explanations.
Correctly aligned verifier reward is causally required.

## Honest boundary and next rung

This is evidence for rapid reward-driven adaptation over genuinely persistent
physical state, not yet evidence for compounding sample efficiency. Next run a
matched persistent-versus-fresh-bank race from the same initialization and
compare:

1. verified-reward area under the curve per verifier bit;
2. verifier bits needed to recover after each utility switch;
3. old-task retention;
4. wall time and physical I/O;
5. shuffled-reward and history-corruption controls.

Keep the first comparison under one minute. Increase bank count or rounds only
if confidence intervals make the matched difference ambiguous.
