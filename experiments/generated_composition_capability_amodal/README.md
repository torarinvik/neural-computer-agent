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
