# Producer→consumer global-parity composition — 2026-08-04

This is the first genuine sequential factor result in the working-memory
pressure test. A previously acquired span-ten complement artifact was loaded
as a producer. A new generic consumer slot had:

- only the producer's 256-wide learned register as input;
- no raw event or recurrent-state input path;
- its own recurrent register for sequence-level accumulation.

The consumer was trained only from attempted actions and deterministic scalar
verifier outcomes on the verifier-private `producer_global_parity` task. The
task keeps the producer's complement cue visible while requiring global parity
of the sequence.

Primary result, span 12, seed `69104`:

- composed controller: `100.00%`;
- parent: `51.17%`;
- producer-only: `49.74%`;
- consumer-only: `50.00%`;
- producer-zeroed: `51.30%`;
- prior-read ablated: `49.48%`.

Replication, seed `69105`, reached `89.58%` composed accuracy with parent,
producer-only, and consumer-only controls at approximately chance. The two
aligned runs therefore span `89.58%`–`100.00%`. Artifact
reload, producer ablation, prior-read ablation, raw-bypass absence, and frozen
core checks all passed in both runs.

This promotes a narrow but important CPU-like result: an external producer
factor can feed a separate recurrent consumer factor, and the pair can learn a
new sequence-level computation that neither factor can execute alone. It does
not establish arbitrary factor algebra, arbitrary program induction, or
general cognition.

Reports:

- `report.json`
- `../prior_only_consumer_global_parity_span12_128_replication_2026-08-04/report.json`
