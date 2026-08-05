# Byte-level grounded text frontier — 2026-08-04

> **Retraction:** The original raw-byte results in this record are invalid as
> capability evidence. A metadata audit found that the caption renderer used
> verifier-generated `context_ids` to choose a shape word. See
> `metadata_leak_audit.json`. The renderer is now pixel-only; all future runs
> start a new qualification record.

## Question

Can a replaceable frontend receive raw variable-length text bytes and make a
frozen amodal controller use a grounded description, without semantic labels,
correct actions, task IDs, or verifier outcomes entering the learner?

## Mechanism

The external caption source described only visible colour, shape, brightness,
and position. The learner received shifted UTF-8 bytes with padding. A small
character n-gram frontend was trained only against the frozen visual encoder's
opaque event tensor. The controller, visual encoder, input bus, and decoder
were frozen and their controller hash was checked after each run.

Styles 0–2 were training paraphrases. Styles 3–4 used held-out syntax and
word order. The causal audit paired the text with the correct partial visual
view, shuffled text across lifetimes, and supplied a contradictory grounded
description. Text-only and vision-only controls were also retained.

## Historical result — retracted

The following numbers are retained for debugging only. They must not be used
as evidence of grounded text capability:

| run | updates | min held-out fused | max shuffled | min contradiction flip | min vision-only |
|---|---:|---:|---:|---:|---:|
| seed 1001001 | 24 | 93.96% | 54.38% | 78.96% | 96.35% |
| seed 1001002 | 32 | 92.19% | 55.73% | 75.63% | 97.08% |
| seed 1001003 | 40 | 90.63% | 53.54% | 74.06% | 96.77% |

The first two runs passed their local gate and the third was a near miss, but
the hidden metadata makes all three results invalid.

The failed byte-GRU rung is retained only as a historical training trace.

## Paired-paraphrase follow-up

The next low-complexity repair showed two different training paraphrases for
each same rendered scene while keeping the visual encoder, controller, decoder,
and held-out gate fixed. It improved the hard diamond fused cells to roughly
91%, but the worst contradiction-flip cell remained `73.0–74.0%` after 32 and
40 updates, below the strict `75%` gate. The arm is therefore not promoted.
The exact summary is in `paired_view_followup_summary.json`. The next repair
should address representation robustness to held-out wording/geometry rather
than spend more updates on this protocol.

The subsequent representation search is bounded in
`representation_frontier_followups.json`: a zero-initialized relative-order
residual, a small byte transformer, and their paired-view combination all
failed their early gates. They are not scaled or promoted. The qualified
character n-gram CNN remains the baseline; the next search should change the
learning/data protocol rather than accumulate more frontend architectures.
The first diamond-replay sampling rung also failed to show early causal signal,
so it is not being scaled.

The repaired pixel-only character n-gram CNN is now the unqualified baseline;
the fresh corrected run is recorded in
`../natural_text_grounding_pixel_only_2026-08-04/`. It passes the three-seed
training-time gate, but its independent saved replay is 2/3, so the corrected
population is not promoted yet.

## Accounting and boundary

- The retracted runs reported no verifier bits or logical lifetimes, but that
  accounting does not repair the hidden metadata leak.
- The 24-update arm used 55,296 paired unlabeled frames; the 32-update arm
  used 73,728; the 40-update arm used 92,160.
- Optimizer updates were 24, 32, and 40 respectively; no examples were
  replayed.
- Controller parameters were unchanged in every saved report.
- Fresh saved-frontend replay was independent of the training optimizer.

No raw UTF-8 grounded-text capability is currently qualified. The next useful
step is a fresh pixel-only three-seed run with the same strict causal gate,
followed only then by a real natural-text source.

Reports:

- `byte_gru_failed_seed1001001.json`
- `byte_cnn_seed1001001_u24.json`
- `byte_cnn_seed1001002_u24_near_miss.json`
- `byte_cnn_seed1001002_u32.json`
- `byte_cnn_seed1001003_u32_near_miss.json`
- `byte_cnn_seed1001003_u40_near_miss.json`
- `saved_frontend_replay_summary.json`
