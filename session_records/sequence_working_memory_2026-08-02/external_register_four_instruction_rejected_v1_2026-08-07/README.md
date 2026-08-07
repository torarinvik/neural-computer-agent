# Four-instruction external-register boundary: rejected diagnostic

Date: 2026-08-07
Status: rejected; diagnostic only
Schema: `neural-computer.external-register-four-instruction-report.v1`

This pressure test extended the read/execute register to a runtime-supplied
four-primitive program:

`reverse -> adjacent_xor -> complement -> prefix_parity`

The controller received only rendered learned events, opaque feedback, and
memory-selected instruction vectors. The four instruction vectors were
acquired sequentially without replay, with the parent and interpreter frozen
after the acquisition stages. A reversed-order program, fresh learner,
reward-shuffled control, missing-evidence control, exact reload,
checksum-corruption, frozen-parent, and stable-prefix gates were included.

## Low-rank canonical operator

At the next acquisition rung (`256` updates per primitive and composition),
reverse, complement, and prefix-parity retention reached `1.0000`, `0.8008`,
and `0.8555`, but adjacent-XOR remained `0.7734`. The inherited composition
reached `0.8477` versus fresh `0.8438`, but failed primitive retention and had
no positive stable transfer. The short rung was also rejected as undertrained.

This isolates the current canonical operator’s expressivity boundary: its
factorized low-rank register transform is composable, but is not yet a robust
nonlinear temporal primitive interpreter.

## Nonlinear candidates

Two structured alternatives were tested without changing the external state
contract:

| Candidate | Primitive retention | Inherited composition | Fresh composition | Decision |
| --- | ---: | ---: | ---: | --- |
| Factorized FiLM | 0.8125 | 0.8633 | 1.0000 | reject; no transfer |
| Low-rank + zero-init FiLM hybrid | 0.9414 | unstable (`0.7734` final) | 1.0000 | reject; no stable composition |
| Hybrid plus 256-update blueprint pretraining | 0.9336 | 0.4805 | 1.0000 | reject; composition collapse |

The hybrid candidates show that better primitive acquisition alone is not
enough. Shared nonlinear pretraining can improve individual instruction
retention while destroying the serial algebra that made the two- and
three-instruction low-rank results transferable. No candidate was promoted,
and no checkpoint was curated.

A composition-aware 64-update blueprint pretraining probe was also rejected:
primitive retention was `0.7813`, inherited composition was `0.7344`, and the
fresh control reached `0.9844`. This did not provide a mechanistic signal for
scaling the curriculum.

The canonical production path therefore remains the factorized low-rank
read/execute interpreter. The next architectural target is a nonlinear
operator with an explicit compositional invariant—likely a typed or gated
intermediate representation—rather than simply adding more hidden layers or
pretraining updates. All reports and accounting are retained here.
