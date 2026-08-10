# Promoted: outcome-only executable-program candidate search

Date: 2026-08-10

Seeds: `23001`, `23002`, `23003`

Schema: `neural-computer.external-program-candidate-search.v1`

## Result

The shared register interpreter is pretrained once and then frozen. One opaque
one-instruction file is protected in external memory. A held-out target file
is a two-instruction composition and is absent from the live file bank. The
new `ExternalProgramCandidateSearch` proposes generic structural edits from an
opaque instruction bank and updates only aggregate operator statistics from
ordered scalar verifier scores.

| seed | warm proposals to target | fresh proposals | warm target mastery | source retention | fresh best score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 23001 | 6 | 256 | 1.0000 | 1.0000 | 0.0573 |
| 23002 | 1 | 256 | 1.0000 | 1.0000 | 0.0739 |
| 23003 | 13 | 256 | 1.0000 | 1.0000 | 0.0463 |

The warm source composes the target by inserting one opaque instruction atom.
The matched fresh control starts from a different atom and cannot reach that
target under the same one-edit budget. Every warm run passes all `17` gates:
target-not-preloaded, exact candidate generation, stable admission through a
`32`-outcome window, protection of old and new files, source and target
mastery, fresh-control rejection, corrupted-file rejection and no-op, shuffled
outcome rejection, exact search-state and file-memory reload, canonical
runtime traversal, frozen interpreter, zero replay, and zero controller
updates.

The candidate generator sees no target artifact, program identity, operation
meaning, task label, or correct action. The private verifier emits only scalar
similarity values. Failed candidates are discarded after aggregate statistics
are updated; only the independently verified candidate is admitted to durable
external memory.

## Claim boundary

This promotes bounded outcome-driven one-edit structural synthesis from a
protected opaque parent. It does not establish multi-step beam search,
arbitrary program induction, Turing-complete learning, unrestricted memory
growth, or general continual learning. The next pressure is a persistent
multi-step candidate frontier and a genuine Brain Workshop family stream.

Reports: `report_seed23001.json`, `report_seed23002.json`,
`report_seed23003.json`.

Reproduce with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_candidate_search/train.py \
  --seed 23001 \
  --report-out /tmp/external-program-candidate-search-23001.json
```
