# Promoted: persistent multi-step executable hypothesis frontier

Date: 2026-08-10

Seeds: `23001`, `23002`, `23003`

Schema: `neural-computer.external-program-hypothesis-frontier.v1`

## Result

The shared register interpreter and controller are frozen. One opaque
one-instruction source file is protected in external memory. A held-out target
is a three-instruction composition and is absent from the live file bank. The
new `ExternalProgramHypothesisFrontier` retains provisional opaque files,
expands them breadth-first with generic edits, and records only aggregate
scalar verifier statistics. The verified target is then admitted through the
existing stable-prefix file transaction.

| seed | warm evaluations | random-parent evaluations | source retention | target mastery |
| ---: | ---: | ---: | ---: | ---: |
| 23001 | 22 | 62 | 1.0000 | 1.0000 |
| 23002 | 28 | 66 | 1.0000 | 1.0000 |
| 23003 | 13 | 50 | 1.0000 | 1.0000 |

Every run passes the target-not-preloaded, exact candidate generation,
protected-root retention, stable admission, corruption no-op, exact frontier
and file reload, canonical runtime traversal, frozen interpreter,
zero-replay, and zero-controller-update gates.

## Claim boundary

This promotes bounded multi-step outcome-only external-memory search and
admission. It does not establish open-ended program induction, unrestricted
memory growth, Turing-complete acquisition, or general continual learning.
The next pressure test is the same frontier on genuinely rendered Brain
Workshop task families with a non-synthetic verifier and held-out retention.

Reports: `report_seed23001.json`, `report_seed23002.json`,
`report_seed23003.json`.

Reproduce with:

```bash
uv run python -m experiments.external_program_hypothesis_frontier.train \
  --seed 23001 \
  --report-out /tmp/external-program-hypothesis-frontier-23001.json
```
