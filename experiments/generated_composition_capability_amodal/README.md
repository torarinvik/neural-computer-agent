# Generated compositional capability pilot

This pilot trains one fresh external capability against a frozen parent
controller on `generated_composition`. The default verifier-private grammar
contains two- and three-primitive programs, while the renderer can also accept
a runtime-supplied grammar of longer programs. Primitive cues are rendered as
ordinary learned events; the controller receives no program ID, primitive
labels, correct actions, or verifier-private composition metadata.

Run the short rung with:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train \
  --parent-updates 32 --updates 64 --batch-size 8 --audit-count 16 \
  --eval-every 16 --report-out /tmp/generated-composition/report.json
```

This is an acquisition diagnostic, not a continual-learning promotion. The
full promotion must append the generated capability behind the protected
route chain and retain every previously mastered artifact.

The serial external-stack diagnostic is:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_pipeline \
  --parent-updates 32 --updates 64 --program-count 2 --batch-size 8 \
  --audit-count 16 --eval-every 16 \
  --report-out /tmp/generated-composition-pipeline/report.json
```

The routed stack uses the new external composition binder:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_pipeline \
  --stack routed --parent-updates 32 --updates 64 --program-count 2 \
  --batch-size 8 --audit-count 16 --eval-every 16 \
  --report-out /tmp/generated-composition-routed/report.json
```

The promoted append-only artifact-bank audit acquires each generated
composition in an isolated external row, protects it with fresh retention
outcomes, and trains an opaque router without updating older artifacts:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 --batch-size 16 --route-batch-size 16 \
  --audit-count 64 --route-audit-count 512 --retention-probes 4 \
  --eval-every 32 --report-out /tmp/generated-composition-artifact-bank/report.json
```

The two-artifact result is promoted only as bounded no-replay growth. It does
not claim general continual learning or open-ended program induction.

The stronger append-only route-chain audit freezes the base route and learns
one extension per new artifact:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 2 --batch-size 16 --route-batch-size 16 \
  --audit-count 64 --route-audit-count 512 --retention-probes 4 \
  --eval-every 32 --report-out /tmp/generated-composition-append-only/report.json
```

This append-only route result is replicated for seeds 69316 and 69317. It is
still bounded growth over the fixed generated grammar.

The next pressure test replaces the fourth family member with composition ID
`6`, a three-primitive `reverse -> complement -> rotate` program. It keeps the
same frozen append-only route chain while testing a longer computation and a
grammar shift:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 2 6 --batch-size 16 --route-batch-size 16 \
  --audit-count 64 --route-audit-count 512 --retention-probes 4 \
  --eval-every 32 --report-out /tmp/generated-composition-artifact-bank-grammar-shift-v1/report.json
```

The grammar-shift result was replicated with seeds `69316` and `69317` and is
archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_append_only_grammar_shift_replicated_promoted_v1_2026-08-06/`.
Both runs acquired the longer artifact, retained all four routes at `1.0000`,
passed cold-start old-route retention and all causal, reload, corruption,
frozen-core, and zero-replay gates, and rejected every stage-specific
reward-shuffled control. This is replicated bounded continual external growth,
not general continual learning.

The renderer and route-key path also accept a runtime-supplied verifier-private
grammar. Repeat `--program-spec` for each custom program, using comma-separated
primitive names:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 2 3 \
  --program-spec forward,reverse,complement,rotate \
  --program-spec rotate,complement,reverse,forward \
  --program-spec complement,rotate,forward,reverse \
  --program-spec reverse,forward,rotate,complement \
  --batch-size 16 --route-batch-size 16 --audit-count 64 \
  --route-audit-count 512 --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-runtime-program/report.json
```

This runtime-grammar result is replicated for seeds `69316` and `69317` in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_grammar_replicated_promoted_v1_2026-08-06/`.
It promotes mechanism transfer to four-primitive runtime programs, while
unrestricted capacity and arbitrary open-ended program induction remain
unqualified.

The transfer pressure test compares an inherited external file with a fresh
candidate on a new runtime program. A stable-prefix verifier selects a unique
winner before the candidate is admitted beside the protected source:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_transfer \
  --parent-updates 128 --source-updates 256 --candidate-updates 256 \
  --batch-size 16 --audit-count 64 --retention-probes 4 --eval-every 32 \
  --program-spec reverse,adjacent_xor,complement,prefix_parity \
  --program-spec prefix_parity,global_parity,rotate,complement \
  --report-out /tmp/generated-composition-transfer/report.json
```

This transfer audit is replicated for seeds `69316` and `69317` in
`session_records/sequence_working_memory_2026-08-02/generated_composition_transfer_replicated_promoted_v1_2026-08-06/`.
The inherited candidate reaches the new target in `6,144` versus `10,240`
fresh bits on seed `69316`, and `4,096` versus `10,240` on seed `69317`.
Both candidates are compared on fresh held-out outcomes; old-file retention,
candidate selection, capacity growth, frozen-core, and zero-replay gates pass.

The next pressure test compares two protected external files, then compacts
them into one physical row with independently addressable opaque views before
growing capacity for the selected target:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_multi_transfer \
  --seed 69316 --parent-updates 128 --source-updates 256 \
  --candidate-updates 256 --batch-size 16 --audit-count 64 \
  --retention-probes 4 --eval-every 32 \
  --source-ids 0 2 --target-id 1 \
  --program-spec reverse,adjacent_xor,complement,prefix_parity \
  --program-spec prefix_parity,global_parity,rotate,complement \
  --program-spec global_parity,reverse,adjacent_xor,rotate \
  --report-out /tmp/generated-composition-multi-transfer/report.json
```

The two-seed promoted audit selected source 0 at `6,144` and `4,096` stable
target bits versus `10,240` fresh bits, saved one physical row while retaining
both source views, and admitted the target by growth after compaction. This is
logical storage compaction rather than neural weight compression, and remains
bounded external transfer rather than general continual learning. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_multi_transfer_replicated_promoted_v1_2026-08-06/`.

The stricter neural-consolidation audit replaces those two separate views with
one shared routed stack. The inherited student starts from source 0 and learns
both source procedures from fresh outcomes; the fresh student is a control. If
the stable-prefix curves tie, inherited weights are retained only when a fresh
maximin verifier gives them a strict worst-source behavior margin:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_distilled_consolidation \
  --seed 69316 --parent-updates 128 --source-updates 256 \
  --consolidation-updates 256 --target-updates 256 \
  --batch-size 16 --audit-count 64 --retention-probes 4 --eval-every 32 \
  --source-ids 0 2 --target-id 1 \
  --program-spec reverse,adjacent_xor,complement,prefix_parity \
  --program-spec prefix_parity,global_parity,rotate,complement \
  --program-spec global_parity,reverse,adjacent_xor,rotate \
  --report-out /tmp/generated-composition-distilled-consolidation/report.json
```

Replicated seeds retained both source procedures in one shared artifact at a
`0.5000` payload ratio, with reloaded source behavior `0.9648/1.0000` and
`0.9922/1.0000`. The consolidated target reloaded at `1.0000` in both runs.
This is behavior-verified neural consolidation, not logical two-view packing,
unrestricted growth, or general continual learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_distilled_consolidation_replicated_promoted_v1_2026-08-06/`.

## Three-source fresh-outcome neural consolidation (2026-08-06)

The next scaling rung replaces three protected external files with one shared
routed stack. The inherited student starts from source 0 and learns source
procedures 0, 2, and 3 from fresh outcomes; a fresh student receives the same
budget. Stable-prefix selection is authoritative, with the strict maximin
behavior margin retained only as the tie fallback.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_distilled_consolidation \
  --seed 69316 --parent-updates 128 --source-updates 256 \
  --consolidation-updates 512 --target-updates 256 \
  --batch-size 16 --audit-count 64 --retention-probes 4 --eval-every 32 \
  --source-ids 0 2 3 --target-id 1 \
  --program-spec reverse,adjacent_xor,complement,prefix_parity \
  --program-spec prefix_parity,global_parity,rotate,complement \
  --program-spec global_parity,reverse,adjacent_xor,rotate \
  --program-spec complement,prefix_parity,reverse,global_parity \
  --report-out /tmp/generated-composition-distilled-consolidation-three/report.json
```

Replicated seeds `69316` and `69317` accepted the shared replacement: three
rows became one, payload ratio was `0.3333`, all aliases resolved to one
artifact digest, and source behavior survived reload at
`0.7578/1.0000/0.8945` and `0.9102/1.0000/1.0000`. The grown target reloaded at
`1.0000` in both runs. The controller stayed frozen and replayed examples were
zero. This is replicated, bounded three-source neural consolidation—not
unrestricted memory growth, arbitrary program induction, or general continual
learning. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_distilled_consolidation_three_source_replicated_promoted_v1_2026-08-06/`.

## Sequential slot-isolated growth across four sources (2026-08-06)

The next pressure test makes the no-replay boundary explicit. Four source
procedures are acquired sequentially. Each new procedure is learned in a
fresh neural slot, then appended under an opaque alias into one physical
artifact row; earlier slots and decoders are never updated. A target is
transferred from the first retained slot and admitted by capacity growth.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_sequential_slot_isolated_consolidation \
  --seed 69316 --parent-updates 128 --source-updates 256 --target-updates 256 \
  --batch-size 16 --audit-count 64 --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-sequential-slot-isolated-four/report.json
```

Replicated seeds `69316` and `69317` accepted all three sequential appends.
The four source aliases shared one physical row and retained behavior after
reload at `0.9570/1.0000/0.9531/1.0000` and
`0.9805/1.0000/0.9844/1.0000`. The inherited target reached stable mastery at
`2,048` fresh bits in both replicas versus `14,336` and `8,192` for matched
fresh controls, and reloaded at `1.0000`. Alias-specific reversal/recovery,
target reversal/recovery, corruption rejection, frozen-core, and zero-replay
gates passed. The four-slot payload ratio was `1.0000`: this is safe
capacity growth, not neural weight compression.

The companion dense shared-weight expansion control was rejected. It learned
the new source at `1.0000` but dropped source 0 from `0.9531` to `0.6250`
without old-source replay, showing that frozen tensors alone do not provide
route isolation. Evidence, accounting, and both rejected controls are in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_slot_isolated_four_source_replicated_promoted_v1_2026-08-06/`.

## Four-source dense expansion with opaque route binding (2026-08-06)

The rejected dense control is now repaired at the external capability
boundary. `ExternalCapabilityComposition.step()` accepts an optional opaque
boolean slot mask supplied by memory. An alias is restricted to the slots
available when it was admitted, while the newly arriving alias can use the
new slot and earlier slots. The controller still receives only learned event
tensors and emits learned intentions; no source ID or verifier grammar enters
the controller.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_sequential_distilled_consolidation \
  --seed 69316 --parent-updates 128 --source-updates 256 \
  --consolidation-updates 512 --target-updates 256 \
  --source-ids 0 2 3 4 --target-id 1 --batch-size 16 \
  --audit-count 64 --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-sequential-route-bound-four/report.json
```

Replicated seeds `69316` and `69317` accepted all three dense append stages.
Final source behavior after reload was `0.9570/1.0000/0.9375/1.0000` and
`0.9805/1.0000/0.9844/1.0000`; inherited target mastery remained `1.0000`
after reload at `2,048` stable verifier bits in both runs, versus fresh
controls at `14,336` and `8,192`. The frozen core, one-row alias identity,
reversal/recovery, corruption, and zero-replay gates all passed. A long
rollout also exposed and fixed exact-zero propensity underflow at the opaque
feedback boundary.

This promotes bounded replay-free dense growth with external route binding.
It does not establish unrestricted memory growth, neural compression,
arbitrary program induction, or general continual learning. The current mask
prevents interference and now skips globally ineligible slots; a matched
four-source follow-up reduced paired wall time from `961.3s` to `831.4s` while
preserving every metric and gate. Batch-divergent masks still execute their
active-slot union. Evidence is archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_sequential_route_bound_dense_four_source_replicated_promoted_v1_2026-08-06/`.

The alias slot binding is now persisted as opaque memory metadata and is
consumed from reloaded `ArtifactHandle` values during retention and behavior
probes. A full seed-`69316` audit reproduced the same gates and metrics. The
write-heavy persistence path took `1,244.6s` versus `831.4s` for the in-memory
mask reference, making batched retention/manifest writes the next bottleneck.
The batch API is now implemented and a focused control reduced eight saves to
one with identical ledger state. The full audit remained semantically exact,
but wall time was noisy (`1,363.9s`), so an end-to-end speedup is not promoted.
Persisted binding metadata is protected by an atomic `manifest.sha256`
sidecar; sidecarless legacy stores remain readable, while tampered bindings
are rejected before execution.

Retention-only updates now persist only `retention-ledger.json`, so they do not
rewrite structural artifact rows, the manifest, or its checksum. The matched
seed-`69316` audit passed the same gates and metrics, but took `1,332.2s`
versus `1,244.6s` for the prior persisted-binding run. This promotes narrower
write scope and crash-boundary clarity, not an end-to-end performance gain;
the negative timing result is archived in
`report_retention_only_persistence_seed69316.json`.

The same runtime grammar can compose nonlocal temporal and aggregation
primitives:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --parent-updates 128 --artifact-updates 256 --route-updates 256 \
  --composition-ids 0 1 2 3 \
  --program-spec reverse,adjacent_xor,complement,prefix_parity \
  --program-spec prefix_parity,global_parity,rotate,complement \
  --program-spec global_parity,reverse,adjacent_xor,rotate \
  --program-spec complement,prefix_parity,reverse,global_parity \
  --batch-size 16 --route-batch-size 16 --audit-count 64 \
  --route-audit-count 512 --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-runtime-nonlocal/report.json
```

This nonlocal runtime-grammar result is replicated for seeds `69316` and
`69317` in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_nonlocal_replicated_promoted_v1_2026-08-06/`.
It promotes a shared compositional interface for temporal and aggregation
procedures, while unrestricted capacity and arbitrary open-ended program
induction remain unqualified.

## Runtime-generated program mechanism transfer (2026-08-07)

The next audit removes the remaining predeclared schedule assumption. A
verifier-private generator samples distinct four-primitive programs at runtime
from the existing primitive registry, rejects functional duplicates of the
fixed grammar, and passes only rendered event streams and scalar outcomes to
the learner. The same frozen controller, isolated artifact blueprint, and
append-only route chain are used for every generated schedule.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --program-seed 1739 --program-count 3 --program-depth 4 \
  --composition-ids 0 1 2 --parent-updates 128 \
  --artifact-updates 256 --route-updates 256 --batch-size 16 \
  --route-batch-size 16 --audit-count 64 --route-audit-count 256 \
  --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-runtime-generated/report.json
```

Across seeds `69316` and `69317`, all three runtime-generated artifacts were
stable and protected, route and permutation accuracy were `1.0000`, the
reward-shuffled controls failed as intended, and reload, corruption,
frozen-core, and zero-replay gates passed. The weakest held-out artifact
behavior was `0.8828`/`0.8906`. This is replicated mechanism transfer beyond a
predeclared append schedule; arbitrary open-ended program induction, learned
compression, and general continual learning remain unqualified. Evidence is
archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_generated_random_replicated_promoted_v1_2026-08-07/`.

## Runtime-generated eight-step programs (2026-08-07)

The same runtime generator now supports eight ordered primitives. Each
primitive is rendered in an ordinal event band, so execution order remains
learnable without a program ID or semantic field:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_artifact_bank \
  --route-mode append_only --base-route-count 1 \
  --program-seed 2718 --program-count 3 --program-depth 8 \
  --composition-ids 0 1 2 --parent-updates 128 \
  --artifact-updates 256 --route-updates 256 --batch-size 16 \
  --route-batch-size 16 --audit-count 64 --route-audit-count 256 \
  --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-runtime-depth8/report.json
```

Both seeds mastered and protected all three runtime-generated eight-step
programs. Route and permutation accuracy were `1.0000`; the weakest held-out
artifact behavior was `0.8711`/`0.9805`; reload, corruption, frozen-core,
shuffled-outcome, and zero-replay gates passed. The short and medium controls
failed only acquisition depth and remain archived as rejected controls. This
promotes a deeper bounded computational interface, not open-ended program
induction or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_artifact_bank_runtime_generated_depth8_replicated_promoted_v1_2026-08-07/`.

## Fresh-rebuild neural consolidation of depth-eight procedures (2026-08-07)

The external routed artifact can now expand by one slot while preserving all
existing slot weights and route slices exactly. A disabled new route keeps the
operation behavior-preserving until fresh outcomes train the added capacity.

With two runtime-generated eight-step source procedures, a fresh three-slot
student won the stable-prefix selector in both seeds. The opt-in fresh-rebuild
policy then required independent retention verification before replacing two
protected rows with one shared artifact:

- source and consolidated reload behavior: `1.0000/1.0000` in both seeds;
- every retention probe: `1.0000`;
- physical rows: `2 -> 1`;
- payload ratio: `0.7392`;
- frozen controller: unchanged;
- replayed examples: `0`.

The fresh rebuild is not counted as inherited positive transfer. Target
transfer is unqualified for this result; the seed-69316 diagnostic target
reloaded at `0.9961`, while seed `69317` did not require a target arm for the
fresh-rebuild promotion. Three-source depth-eight controls failed retention at
both 256 and 512 consolidation updates, so the next bottleneck is
multi-procedure credit assignment/capacity selection, not basic artifact
storage.

Evidence and rejected controls are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_depth8_fresh_rebuild_consolidation_replicated_promoted_v1_2026-08-07/`.

## Replay-free staged growth for three depth-eight procedures (2026-08-07)

The historical staged external-growth path handles three runtime-generated
eight-step procedures. It froze old slot parameters and routing state, but
its all-slot masks allowed a new source to use old outputs; therefore it is
not the authoritative isolated-slot result. Every stage still required fresh
retention probes for the new and earlier aliases before adoption.

Across seeds `69316` and `69317`, both stages were adopted. All reload,
reversal/recovery, corruption, frozen-core, and zero-replay controls passed;
the target reloaded at `1.0000` in both seeds. This remains a bounded
retention result, but it does not establish isolated fresh-capacity
acquisition, arbitrary new computation, unrestricted growth, or general
continual learning. The strict correction is documented below.

Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_depth8_sequential_slot_growth_three_source_replicated_promoted_v1_2026-08-07/`.

## Runtime-generated opaque operators (2026-08-07)

The next audit removes the fixed named-primitive registry from the generated
family. With `--primitive-family opaque_rule`, the verifier samples from all
256 eight-bit functions over a three-cell binary neighborhood and renders each
rule as a generic barcode in its ordinal event band. The controller receives
only the rendered event sequence and verifier outcome; rule tokens, rule
tables, and composition IDs remain outside the deployed learner.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_sequential_distilled_consolidation \
  --program-seed 4242 --primitive-family opaque_rule \
  --program-count 4 --program-depth 8 --source-ids 0 1 2 --target-id 3 \
  --parent-updates 128 --source-updates 256 --consolidation-updates 512 \
  --target-updates 256 --batch-size 16 --audit-count 64 \
  --retention-probes 4 --eval-every 32 \
  --report-out /tmp/generated-composition-opaque-rule/report.json
```

The first run used an all-slot acquisition mask, so it did not prove that a
new procedure learned in fresh capacity. The corrected rerun binds each new
source to its newly appended slot while old aliases retain their previous
bindings. Across seeds `69316` and `69317`, both strict staged acquisitions
adopted; source reload behavior was `1.0000/1.0000/1.0000` and
`1.0000/0.9844/1.0000`, target reload was `1.0000` in both, and all
reversal/recovery, corruption, frozen-core, exact-reload, and zero-replay
gates passed. It does not yet establish arbitrary program induction, learned
compression, unrestricted memory growth, or general continual learning.
Evidence and accounting are in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_three_source_strict_isolated_replicated_promoted_v1_2026-08-07/`.

## Runtime-generated strict growth to four procedures (2026-08-07)

The strict isolated-slot trainer now supports verifier-side runtime grammar
generation directly. With `--program-seed 4242 --primitive-family opaque_rule`
and five generated depth-eight procedures, both seeds adopted all three staged
source additions and admitted the fifth target into a grown physical row.
Source reload behavior was `1.0000/0.9961/1.0000/0.9766` and
`1.0000/0.9648/1.0000/0.9844`; target reload behavior was `0.8750` and
`1.0000`. Frozen-core, exact-reload, reversal/recovery, corruption, and
zero-replay gates passed.

This removes the manual program-specification gap, but does not change the
claim boundary: the result is bounded isolated external growth over a finite
opaque-rule family, not arbitrary program induction or general continual
learning. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_four_source_runtime_generated_replicated_promoted_v1_2026-08-07/`.

## Runtime-generated strict growth to five procedures (2026-08-07)

The same runtime-generated opaque family now passes four sequential staged
additions. With six generated depth-eight procedures, both seeds retained all
five source aliases after reload and admitted the sixth target into a new
physical row at `1.0000`. Source reload behavior was
`1.0000/0.9961/1.0000/0.9844/0.9375` and
`1.0000/0.9844/1.0000/0.9883/0.9375`; frozen-core, reversal/recovery,
corruption, exact-reload, and zero-replay gates passed.

The final isolated payload was `1,677,280` bytes, confirming linear external
growth rather than unverified weight sharing. This is a bounded capacity
result; routing under an unbounded bank, learned compression, arbitrary new
computation, and general continual learning remain open. Evidence and
accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_five_source_runtime_generated_replicated_promoted_v1_2026-08-07/`.

## Strict isolated growth to four opaque procedures (2026-08-07)

The strict slot-binding correction now scales from three to four
verifier-private eight-step procedures. Across seeds `69316` and `69317`, all
three staged additions adopted, every source alias retained after reload, and
the fifth target procedure was admitted into a newly grown physical row and
reloaded at `1.0000`. Source reload behavior was
`1.0000/0.9961/1.0000/0.9766` and `1.0000/0.9648/1.0000/0.9844`.

The frozen controller digest, corruption, reversal/recovery, exact-reload,
and zero-replay gates all passed. The short curriculum control failed
mastery, while a 1,024-update shared consolidation control still lost the
third procedure. This confirms that isolated append-only capacity is currently
the reliable retention mechanism; shared consolidation remains an open
credit-assignment problem.

Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_four_source_strict_isolated_replicated_promoted_v1_2026-08-07/`.

## Learned routing across six growing physical rows (2026-08-07)

The append-only route chain now addresses six runtime-generated opaque
artifacts stored in six independently protected rows. Across seeds `69316`
and `69317`, route accuracy, key-permutation accuracy, and reloaded route
accuracy were all `1.0000`; reward-shuffled routing was `0.0000`, and every
artifact remained mastered. The controller stayed frozen and replay remained
zero.

The short control was safely rejected when the capacity planner would have
needed to evict an unprotected row. The trainer now records this partial
failure and its accounting instead of crashing. This is the next verified
address-discovery boundary, not unbounded routing or general continual
learning. Evidence and accounting are archived in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_append_only_route_six_runtime_generated_replicated_promoted_v1_2026-08-07/`.

## Strict isolated growth to ten opaque procedures (2026-08-07)

The slot-isolated trainer now scales the runtime-generated opaque-rule family
to nine sequential fresh source additions plus a tenth target procedure.
Across seeds `69316` and `69317`, every append adopted, all nine source aliases
remained protected after reload, the target grew into a separate row at
`1.0000`, and reversal/recovery, checksum corruption, frozen-core, exact
reload, and zero-replay gates passed. The weakest final source behaviors were
`0.9141` and `0.8320`.

This is the largest strict isolated opaque-procedure result so far, but its
payload grows linearly with one slot per procedure. Inherited target transfer
was not promoted: it tied fresh on one seed and lost on the other. Shared
computation, learned compression, arbitrary program induction, and general
continual learning remain open. Evidence, including the hard-gate short
control, is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_opaque_rule_sequential_slot_growth_ten_source_runtime_generated_replicated_promoted_v1_2026-08-07/`.

## Shared-base plus isolated residual growth (2026-08-07)

`train_shared_residual_bank` tests a smaller external memory contract: one
shared context encoder, one isolated residual slot per acquired procedure, and
one opaque decoder per slot. The first slot trains the shared basis; later
slots train only their own residual and decoder after the shared basis and old
slots are frozen.

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_shared_residual_bank \
  --program-seed 4242 --program-count 3 --primitive-family registry \
  --source-ids 0 1 --parent-updates 64 --slot-updates 128 \
  --batch-size 16 --audit-count 64 --retention-probes 4 \
  --eval-every 32 --torch-threads 1 \
  --report-out /tmp/shared-residual/report.json
```

Two seeds promote the related registry pair with exact reload, old-slot
retention, frozen-core equality, zero replay, and a `0.5556` parameter ratio
versus independent full programs. The same budget rejects two genuinely
opaque procedures at `0.6250`, and a third heterogeneous registry procedure
at `0.5430`. This is verified shared computation over a common learned basis,
not arbitrary new computation or general continual learning. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_bank_registry_replicated_boundary_v1_2026-08-07/`.

## Append-only residual compute for opaque rules (2026-08-07)

The `--residual-compute` variant gives each appended slot a compact local
recurrent compute encoder in addition to its intention adapter. With local
hidden/width `32/16` and 256 updates, the same opaque pair that rejected the
adapter-only bank now promotes at both seeds: the new slot reloads at
`0.8906/0.9023`, the old slot remains at `1.0000`, and corruption recovery,
exact reload, frozen-base, frozen-old-slot, frozen-core, and zero-replay gates
pass. This is bounded append-only compute growth, not arbitrary open-ended
computation. Evidence is in
`session_records/sequence_working_memory_2026-08-02/generated_composition_shared_residual_compute_opaque_replicated_promoted_v1_2026-08-07/`.
