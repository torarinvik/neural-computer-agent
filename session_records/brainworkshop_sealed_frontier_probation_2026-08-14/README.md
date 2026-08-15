# Sealed frontier probation

This record is **probation**, not a promotion. It is one seed per arm and
does not consume a one-use holdout. It does close four previously open
holes on the live Position N-Back line.

## What was tested

1. The relative-address primitive is discovered by outcome-only one-hot
   search on live Neural Workshop 1-back. It is not loaded as a gifted
   `PREVIOUS` file.
2. Unknown 2-back is resolved by the autonomous header policy: the
   discovered one-step file fails, then `PREVIOUS ∘ PREVIOUS` is composed
   and admitted.
3. Five-back composition misses at `max_history=4`. History is then grown
   to 8 as a versioned interpreter with unchanged relation weights, and
   five-back verifies.
4. The same primitive and its two-step composition execute on rendered
   audio, a second substrate.
5. Dual N-Back uses the same two-way decoder once per source and packs
   bits. Seed 96017 scored `1.000` on Dual 1-Back and Dual 2-Back. A
   one-step program on 2-back scored `0.261`.

A related one-seed founding comparison (header variants of a known depth)
is archived beside this report as `founding_report.json`.

## Seed-95017 sealed frontier

| Arm | Result | Bits |
| --- | --- | ---: |
| Discover offset 0 on 1-back | 1.000 | 35 |
| Autonomous 2-back compose | 1.000 | 32 |
| 5-back at history 4 | capacity miss | 0 |
| 5-back after grow to 8 | 0.952 | 21 |
| Warm audio 1-back | 1.000 | 59 |
| Warm audio 2-back | 1.000 | 58 |
| Fresh audio 1-back discover | offset 0 at 1.000 | 59 |

Controller, program, and replay updates were zero. The grown controller
digest differs because `max_history` is part of the interpreter identity;
relation weights were copied unchanged.

## Seed-94017-v2 founding (header transfer)

Warm 3-cell 3-back retrieved by the same-slot invariant at 20 bits versus
85 fresh bits (`4.25×`). First-time 2-cell 3-back was a tie (117 vs 119).
Source 1/2/3-back retained `1.000/1.000/0.943`.

## Why this is not promoted

- one seed per campaign;
- no external promotion population or holdout lease;
- Dual N-Back on the physical Neural Workshop still needs two public ports;
- first-time depth invention is not cheaper than a matched climb.

Retain the blueprint. Do not treat an isolated threshold as mastery.
