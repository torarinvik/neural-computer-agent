# Session export — Neural Workshop induction path (2026-08-15)

This is a stand-alone brief of the continued Neural Computer / Neural Workshop
session. It is not a promotion record and does not claim open program
induction.

## Frozen identities

| Item | Value |
| --- | --- |
| Controller file SHA-256 | `93a4dbb72953c60b7964546490c202ba9c6f4a4f9b4289de2c3f7db986690537` |
| Controller digest | `59c9ef2b235104e4f0d6bc143ba195fb57a907da9f29b1d5750c39fa22f7687c` |
| `AgentBrain.bank` file SHA-256 | `07319eb13c9cac58bbbe94258548e64f13ba1b3512ca2a01fbdc675c49e2e7c9` |
| Slot 0 | Position 1-back `90e20193…` (do not rewrite) |
| Slot 1 | Gym Dual 1-back |
| Slot 2 | compose(0, 0) depth-2 child |
| Curated frontend | `artifacts/checkpoints/rendered_frontend_seed1001.pt` |
| Frontend content digest | `1ce405a091b7b0bac7ee4e2c96d027b81188f70baa31db2601057222c0f74b5d` |
| Frontend file SHA-256 | `751a9f377d732e78dd0dbacff7bc1c70d59d3135e660f31241b1be4d6e0c6807` |

Packed Dual actions are a decoder adapter. They do not mint a second
controller. Desktop Dual is I/O, not a trainer. Doom and a semantic teacher
are deferred.

## Architecture (unchanged)

`N encoders → amodal event bus → one controller/memory → intention bus → M decoders`.

The controller consumes learned events and emits intentions. Latent meanings
are not assigned by hand. Learner-visible streams: rendered vision/audio/text,
own actions, own state, deterministic scalar verifier outcomes.

## Search grammar (closed, larger than delay-only)

Order: retrieve → same-primitive compose → invert → and(invert(delay), prototype) → invent last.

| Kind | Meaning |
| --- | --- |
| retrieve | Load an admitted file |
| compose | Repeat one equal one-row temporal-address primitive |
| invert | Flip the intention of a delay/prototype file (`neural-computer.intention-invert.v1`) |
| and | Invert(delay) and a prototype on the same tick (`neural-computer.intention-and.v1`) |
| invent | Fresh prototype-match template, always last |

Unequal delay primitives still fail closed. Double invert fails closed.
Prototype and AND admits require a frontend digest. Unbound prototypes cannot
enter the bank.

## What was measured

### Prototype-match (non-delay operator)

Compares the current event to a stored event-width template. Delay files fail
current-symbol. Invent zeros fail frozen. Action-level credit updates only
`prototype` under that schema.

Unused random-encoder acquire (116017–118017): hold 1.000 / 48 bits; zeros,
reverse, shuffle, delay slot 0 below threshold. Status
`replicated_not_admitted`. Record:
`session_records/brainworkshop_current_symbol_acquire_2026-08-15/`.

### Bound frontend

Unused seeds 119017–121017 on `rendered_frontend_seed1001.pt`: hold 1.000 / 48.
Cross-encoder stays at chance. Same learned template transferred 1.000 on
119018, 120017, 120018, 121017, 121018 with zero updates. Status
`replicated_not_admitted`. Record:
`session_records/brainworkshop_current_symbol_bound_frontend_2026-08-15/`.

### Search invent lease

Unused seeds 122017–124017: delay retrieves failed, invent acquired and bound
the frontend, six frozen holds at 1.000 (288 bits; first stable prefix 48).
Status `replicated_not_admitted`. Record:
`session_records/brainworkshop_current_symbol_search_lease_2026-08-15/`.

### Onset lease (AND, unused seeds)

Unused seeds 125017-127017 at 48 steps: search selected `and:0` on every seed
and held `1.000` for six frozen sessions (282 bits, stable prefix 47), but the
prototype-only control reached `0.830` on 127017. Status `rejected`. Record:
`session_records/brainworkshop_onset_search_lease_2026-08-15/`.

Episode length was the only axis changed. Fresh block 128017-130017 at 192
steps: `and:0` on every seed, `1.000` × 6 (1146 bits, stable prefix 191), with
retrieve slot 0 at `0.251`, invert slot 0 at `0.749`, prototype-only at
`0.707`-`0.759`, reversed `0.000`, reward-shuffled `0.49`-`0.51`, cross-encoder
`0.749`. Status `replicated_not_admitted`. Record:
`session_records/brainworkshop_onset_search_lease_long_2026-08-15/`.

Neither campaign writes `AgentBrain.bank`. The prototype-capable machine cannot
install recursive depth-2 files, so search covered retrieve {0,1}, invert {0,1},
`and`, and invent; retrieve slot 2, both composes, and invert slot 2 were
proposed but not executed.

### Stale record: current-symbol search lease

The recorded 122017-124017 lease claims `invent` on every seed, but that record
predates `and` entering the grammar ahead of `invent`. Re-running it under the
current code at 48 steps gives `and` at `0.812` on 122017 and 123017 and status
`rejected`; `tests/test_current_symbol_acquire.py::test_search_lease_binds_frontend_and_does_not_write_the_bank`
fails for the same reason at 24 steps. Diagnostic re-runs at 96 and 192 steps
return `invent` at `1.000` on all three seeds. The threshold, not the grammar,
is the weak part: `0.8` over ≤48 eligible trials does not separate these
families. The committed record and that test have not been rewritten; that is a
decision to make explicitly.

### Changed-symbol

Public rule: press when the current symbol is not the previous one. Retrieve
of slot 0 stays below 0.8. Invert of slot 0 clears 0.8. After admitting
invert(0) on a temp bank, later search retrieves that child.

### Onset (needs two families)

Public rule: press when the current symbol is the target **and** it just
changed. Retrieve and invert alone stay below 0.8. AND works if the prototype
is a current-symbol template, or after two-phase acquire: act as invert so
verifier rewards label change trials, then set the prototype to the running
mean of invert-match + reward-1 events. Search winner is `and`. An AND child
can be admitted on a temp bank with parent lineage. Not written to
`AgentBrain.bank`.

## Dual (separate, already promoted)

Gym Dual 1-back holdout (113017–115017) is consumed. Warm Dual 2-back is
composed execution with zero target updates, not a bits-to-threshold transfer
ratio. Desktop Dual `--search` selects a file then executes; `--mode learn` is
refused.

## Not claimed

- Open program induction
- Prototype or AND slot in curated `AgentBrain.bank`
- Dual 2-back bits-to-threshold transfer ratio
- Desktop Dual as a trainer
- Learned proposer over the grammar
- Loops / new operator types beyond this catalog
- Doom or a semantic/world-model teacher

## Key files

- `experiments/brainworkshop_canonical/program_search.py`
- `experiments/brainworkshop_canonical/bank_program.py`
- `experiments/brainworkshop_canonical/current_symbol_acquire.py`
- `experiments/brainworkshop_canonical/rendered_live.py`
- `experiments/brainworkshop_canonical/rendered_environment.py`
- `src/neural_computer/temporal_program.py`
- `src/neural_computer/program.py` (`frontend_digest`)
- `docs/AMODAL_N_TO_M_ARCHITECTURE.md`

## Next (in order)

1. Decide the episode-length floor for lease acceptance, then re-run or retire
   the stale current-symbol search-lease record and its failing test.
2. A proposer that ranks invert/and/invent instead of full enumeration.
3. Dual 2-back same-task learn transfer on unused seeds, if that axis is next.
4. Doom / semantic teacher only much later, and only with authenticated public
   scalars — never privileged semantic fields.

TUI note: `/export` in the Grok TUI also copies or writes the raw conversation.
This file is the scientific brief of the same session.
