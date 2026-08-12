# Promoted archive scale, reversal, and integrity pressure test

This record promotes the next memory-substrate rung after bounded active-cache
retention. `EpisodicBindingArchive` v2 stores 1,024 opaque binding records
behind four active slots while the canonical controller and learned event
encoder remain frozen.

## Replicated result

Seeds 17 and 18 both passed:

- `1.0000` known-binding batch retrieval accuracy over 512 fresh queries;
- `0.0000` unknown false-known rate;
- `1.0000` query-order permutation accuracy and scalar/batch consistency;
- `1.0000` reload retrieval with active residency preserved;
- scalar reversal demoted the stale protected record after four failures while
  its protected sibling survived;
- finite JSON and compact tensor-snapshot corruption were rejected by their
  canonical SHA-256 checksums;
- frozen controller, frozen event encoder, and zero replay.

The cached matrix path processed about 295k queries/second in this local
pressure run. The JSON payload was approximately 645 KB, while the compact
tensor snapshot was approximately 166 KB per 1,024-record archive. Compact
reload preserved retrieval exactly. This measures retrieval and integrity, not
learned capability acquisition or semantic compression quality.

Reports and the accounting ledger:

- `report_seed17.json`
- `report_seed18.json`
- `sample_efficiency_ledger.json`

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_archive_scale_reversal \
  --seed 17 --report-out /tmp/archive-scale.json
```

## Claim boundary

This promotes a scalable, checksummed, reversal-aware external archive
primitive. It does not claim that the frozen controller learned 1,024
capabilities, that arbitrary compression is solved, or that general
continual learning is solved.
