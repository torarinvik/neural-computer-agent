# Byte-level grounded text alignment

This diagnostic is the next language boundary after the archived fixed
word-ID caption adapter. It lowers variable English-like descriptions to
shifted UTF-8 bytes, then trains a small replaceable byte frontend to match a
frozen visual encoder's opaque neural-IR event. The text source sees only the
rendered pixels and visible object properties.

Training uses paired encoded-event consistency only. It does not use verifier
outcomes, correct actions, task IDs, or semantic target vectors. Styles 0–2
are training paraphrases; styles 3–4 are held out for the causal audit.

This experiment can qualify a surface-language transport bridge if the held-out
styles compose with vision, shuffled text collapses, contradictory text flips
the prediction, and the frozen controller plus vision-only behavior are
unchanged. It is still a synthetic grounded-language result, not a pretrained
LLM, natural speech, or open-world language claim.

Run the focused tests with:

```sh
uv run pytest -q experiments/natural_text_grounding/test_train.py
```

Run a short alignment from the existing frozen amodal controller with:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --report /tmp/amodal_byte_text.json \
  --adapter-out /tmp/amodal_byte_text.pt
```

The next stability arm keeps the frontend and controller fixed while showing
two different training paraphrases for each same rendered scene:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --paired-text-views \
  --report /tmp/amodal_byte_text_paired.json \
  --adapter-out /tmp/amodal_byte_text_paired.pt
```

This is still paired encoded-event consistency, not a semantic label or
verifier signal. It is a training-protocol experiment and must earn a new
population gate independently.

The next representation arm adds four content-relative order-pooling bins to
the same byte n-gram frontend:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --position-bins 4 \
  --report /tmp/amodal_byte_text_order.json \
  --adapter-out /tmp/amodal_byte_text_order.pt
```

The bins are transport-level sequence features, not hand-written semantic
fields. The controller remains frozen and receives only the resulting opaque
event tensor.

If the residual CNN arm remains below the gate, the sequence-aware comparison
is available as a separate diagnostic:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --frontend transformer \
  --report /tmp/amodal_byte_text_transformer.json \
  --adapter-out /tmp/amodal_byte_text_transformer.pt
```

This remains a replaceable frontend experiment. It does not create a second
controller or expose text tokens to the cognitive core.

## Corrected pixel-only baseline

The first raw-byte record was retracted after a metadata audit found that its
caption renderer used verifier-generated context IDs. The current renderer is
pixel-only. The corrected character n-gram CNN passed the training-time gate
for three seeds, and optimizer-free 1,024-lifetime replay passed all three
seeds. The synthetic pixel-grounded UTF-8 event transport bridge is promoted;
it remains distinct from open-ended natural-language understanding. The
evidence is recorded in
`session_records/natural_text_grounding_pixel_only_2026-08-04/`.

Replay a saved frontend independently with:

```sh
uv run python -m experiments.natural_text_grounding.audit_saved_frontend \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --adapter artifacts/checkpoints/amodal_pixel_only_text_frontend_seed1001002_u16.pt \
  --report /tmp/amodal_byte_text_saved_replay.json \
  --seed 1001002 --count 256
```

The current data-protocol candidate keeps the qualified CNN and paired-view
loss, but repeats the diamond appearance once in a fixed four-batch training
cycle:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --paired-text-views --diamond-replay \
  --report /tmp/amodal_byte_text_diamond_replay.json \
  --adapter-out /tmp/amodal_byte_text_diamond_replay.pt
```

The held-out styles remain styles 3–4; only the visible training appearance
mixture changes. This is a trainer-only sampling protocol, not a new model
branch.

Generated reports and checkpoints belong in a session record only after the
promotion gate is met; disposable outputs stay outside Git.

## Independent external-caption corpus v2

The promoted next boundary uses a separate versioned phrase corpus with two
authored training variants per style. The frontend, controller, and causal
gate are unchanged:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --text-source external_corpus_v2 \
  --seed 1002001 --updates 16 --batch-size 128 --eval-count 96 \
  --report /tmp/external_caption_v2.json \
  --adapter-out /tmp/external_caption_v2.pt
```

All three v2 saved frontends pass the 1,024-lifetime replay. The promotion
record and exact corpus hash are in
`session_records/natural_text_grounding_external_corpus_v2_2026-08-04/`.

## Static pre-authored annotation table v3

The next boundary removes runtime phrase-slot filling. The source table stores
complete captions for each visible scene key; the runtime sees pixels only to
join an image to its static annotation row and then emits padded UTF-8 bytes.
The table contains no verifier IDs, semantic labels, or format placeholders.

Run the promoted training configuration with:

```sh
uv run python -m experiments.natural_text_grounding.train \
  --controller artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt \
  --input-bus artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt \
  --text-source external_annotation_table_v3 \
  --seed 1003001 --updates 25 --batch-size 128 --eval-count 96 \
  --report /tmp/external_annotation_v3.json \
  --adapter-out /tmp/external_annotation_v3.pt
```

All three saved frontends pass the optimizer-free 1,024-lifetime replay. The
promotion record, accounting, rejected-repair history, and exact annotation
table hash are in
`session_records/natural_text_grounding_external_annotation_table_v3_2026-08-04/`.
This establishes static pre-authored caption transport, not open-world
language understanding, speech, or semantic reasoning.
