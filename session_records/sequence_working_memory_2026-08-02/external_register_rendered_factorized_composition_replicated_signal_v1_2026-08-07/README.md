# Rendered factorized external-register composition: replicated signal

Date: 2026-08-07  
Status: positive replicated composition signal; not promoted as transfer  
Schema: `neural-computer.external-register-rendered-composition-report.v1`

The factorized register machine was tested on valid rendered sequence-memory
events after adding a recurrent external context that ingests every active
event. The parent controller was trained first and then frozen. Reverse and
complement were acquired as separate instruction data with separate external
decoders. The machine and instructions were frozen before a fresh decoder
learned the held-out `complement_reverse` composition.

Both seeds retained reverse after the second instruction (`0.9844` and
`0.9688`) and reached composition accuracy `0.9844` and `0.9805`. The
reward-shuffled composition arms fell to `0.4336` and `0.2891`, while matched
fresh composition learners reached `0.9492` and `0.8750`. Exact machine and
decoder reloads reproduced composition behavior, and the frozen parent digest
was unchanged in both runs. No examples were replayed.

This is a meaningful mechanistic boundary result: an isolated external
recurrent context can ingest a real event stream, factorized instructions can
compose through one register, and old output behavior remains retained. It is
not yet a promoted sample-efficiency or general continual-learning result.
The runs did not measure stable-prefix bits to threshold and did not include
missing-evidence, memory-corruption, or a full valid-pixel fresh control. The
next rung must add those gates and report learning curves, not just final
accuracy, before claiming positive transfer.
