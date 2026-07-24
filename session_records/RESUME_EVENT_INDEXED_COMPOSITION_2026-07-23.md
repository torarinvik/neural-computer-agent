# Resume point: event-indexed composition

## Preserved result

The audited two-event temporal loop is solved with the supervised-bootstrapped
factorized router. The reusable content-addressed event reader is independently
audited at 71--75% on untouched lifetime-disjoint tests, including a
mixed-visual-surface model at 73.5%.

The best frozen four-color composition result is currently 33.98%, 35.94%,
37.89%, and 36.72% at zero, one, two, and four shots. This is partial mapping
transfer, not full compositional mastery.

## Architecture state

- Active memory: consolidated state plus support-rule sidecar.
- Event archive: immutable sensory-derived study writes.
- Content reader: selects a study event from query content.
- Factorized router: receives the reader's first/second candidate actions and
  composes them with the demonstrated temporal rule.
- No game-state or verifier information enters the agent.

## Exact next experiment

Run the prepared `probe_temporal_rule_memory.py` compositional mode at the raw
write boundary. Start with a sub-minute smoke test, then at most 512
train/held-out lifetimes if the pipeline and shuffled-label control are healthy.

Interpretation:

- Rule decodable from raw writes: repair aggregation/read routing.
- Rule absent from raw writes: localize which support event or consolidation
  boundary loses it before building anything.

Do not begin a large behavioral run until this probe identifies the boundary.

## Canonical artifacts

- `experiments/forward_transfer_attention/remote_results_2026-07-23/factorized_router_cache1024_agent_s1_ood5.pt`
- `experiments/forward_transfer_attention/remote_results_2026-07-23/final_gated_audit/`
- `experiments/forward_transfer_attention/remote_results_2026-07-23/event_indexed_reader/content_addressed_reader_sensory1024_val256_test512_seed653.pt`
- `experiments/forward_transfer_attention/remote_results_2026-07-23/event_indexed_reader/content_reader_mixed_surface1024_seed719.pt`
- `experiments/forward_transfer_attention/remote_results_2026-07-23/event_indexed_reader/agent_content_mixed_surface_s1.pt`
- `experiments/forward_transfer_attention/remote_results_2026-07-23/event_indexed_reader/remote_bundle/`
