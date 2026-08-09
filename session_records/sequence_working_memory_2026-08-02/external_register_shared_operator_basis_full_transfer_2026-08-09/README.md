# Shared operator basis: full transfer screen

Date: 2026-08-09

Seeds: `69316`, `69317`

Mode: `factorized_shared_operator_basis`

Operator rank: `8`

Shared basis count: `4`

Source updates: `576`

Target updates: `512` per target

Composition targets: two held-out three-instruction programs
Replay: `0`

## Question

Does a learned opaque instruction code select a mixture of common low-rank
transition factors, allowing the frozen external-register interpreter to learn
new primitive and composed programs while preserving the mastered sources?

This is a portable-computation intervention at the controller/interpreter
boundary. It does not expose task names, protocol actions, positions, correct
answers, or verifier-private metadata. The mode is opt-in; the existing
factorized low-rank mode remains the default.

## Result

The behavior and retention result replicated:

- source mastery passed on both seeds;
- all `8/8` direct and composed targets passed the stable target gate;
- all source retention deltas were exactly zero;
- reward-shuffled and missing-evidence controls remained near chance;
- persistence, corruption, and frozen-parent controls passed;
- the audit's narrow behavior gate promoted on both seeds.

Fresh transfer did not replicate as a general accelerator. Only `1/4`
held-out composition comparisons showed positive transfer under the strict
stable-prefix comparison (`1/2` on seed `69316`, `0/2` on seed `69317`). The
operator basis therefore earns retention as a qualified shared-computation
direction, but not a claim of general continual-learning transfer.

Per-seed accounting was identical: `1,067,008` unique verifier bits,
`8,160` optimizer updates, `139,008` unique logical lifetimes, and no replayed
examples.

## Decision

Retain the opt-in operator basis and its tests as a qualified architectural
candidate. Do not make it the default and do not claim that it solves general
continual learning. The next bottleneck is a portable execution-state algebra
that turns shared factors into reliable learning acceleration on genuinely new
programs and longer compositions, followed by no-replay sequential acquisition
with fresh-task transfer controls.

The durable summaries are in `report_seed69316.json` and
`report_seed69317.json`. The disposable full reports were emitted under
`/tmp/neural-computer-next/` during the run.
