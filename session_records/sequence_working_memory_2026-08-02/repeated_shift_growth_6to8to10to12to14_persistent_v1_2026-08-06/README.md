# Persistent isolated 46-capability growth (2026-08-06)

This is the persistence follow-up to the promoted isolated fourth-shift
growth frontier. The same frozen-controller schedule grows from length-six
through length-eight, ten, twelve, and fourteen episodes to 46 capabilities.
All route extensions and credit heads are external memory-side modules. The
harness writes 89 independently checksummed state files per seed, reloads
them into fresh module instances, reruns route and credit audits, and probes a
deliberately corrupted state copy. Controller state remains frozen and no
training examples are replayed during persistence.

| gate | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| four-shift acquisition/retention gates | pass | pass |
| reloaded old route | 1.0000 | 1.0000 |
| reloaded combined credit | 0.9348 | 0.9130 |
| persistent route reload | pass | pass |
| persistent credit reload | pass | pass |
| corruption rejection | pass | pass |
| persisted state files | 89 | 89 |
| persistence verifier bits / optimizer updates / replay | 0 / 0 / 0 | 0 / 0 / 0 |

The state itself remains replaceable and controller-independent. The reports
record every state digest; disposable state snapshots are intentionally not
committed as checkpoints. This promotes a durable external route/credit state
boundary, not unbounded growth, arbitrary program induction, positive
transfer, or general continual learning.

Reports and the accounting ledger are in this directory; `SHA256SUMS`
verifies the archived evidence.
