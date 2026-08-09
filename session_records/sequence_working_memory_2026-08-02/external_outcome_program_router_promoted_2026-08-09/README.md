# Outcome-only routing over executable external programs

Date: 2026-08-09; seeds: `69316`, `69317`

This audit closes the next seam after delayed-credit promotion. A shared
external register interpreter executes three pre-admitted opaque program
artifacts. A memory-side router samples a two-phase program sequence and
receives only one terminal scalar verifier outcome. The third artifact is
activated append-only between source and target acquisition; the source and
target router states are separate external states. The controller, interpreter,
program memory, router rule, and value-baseline rule are frozen during route
acquisition.

## Result

| seed | source mastery | target accuracy | stable target episodes | no-trace | shuffled | no-append capacity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.9800 | 0.9100 | 8,500 | 0.4300 | 0.0733 | 0.5767 |
| 69317 | 0.9767 | 0.9300 | 3,500 | 0.2100 | 0.1300 | 0.3967 |

Both seeds passed source mastery and retention, target stable-prefix mastery,
appended-program use, no-trace rejection, reward-shuffle rejection,
no-append capacity rejection, missing-feedback no-write, router and artifact
persistence, frozen executor/program memory/rules, and zero replay. Each seed
used `2,000` source and `9,000` target route lifetimes, accounting for
`11,000` unique verifier bits and zero router optimizer updates. The shared
interpreter was pre-admitted in a separate `900`-update external pretraining
fixture; those updates are reported separately and are not claimed as
outcome-only route learning.

## Claim boundary

This promotes a bounded bridge from scalar delayed credit to execution of
opaque external programs: memory can acquire a new multi-step capability by
selecting and composing an appended executable artifact while preserving an
older capability state. It does not prove program induction, arbitrary new
computation, unrestricted memory growth, or general continual learning. The
next pressure test is many-capability interference with one growing router and
then fresh-relation transfer against a matched fresh executor.

The reproducer is
`experiments/external_outcome_program_router/train.py`. Per-seed summaries are
in `report_seed69316.json` and `report_seed69317.json`.
