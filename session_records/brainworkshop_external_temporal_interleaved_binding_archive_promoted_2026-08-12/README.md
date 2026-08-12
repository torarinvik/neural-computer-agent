# Promoted repeated interleaved external binding archive

This record promotes the next bounded continual-learning rung for the frozen
amodal controller. Four anonymous learned-event bindings are introduced across
two physical active-cache slots and revisited through six admission/replacement
cycles. The long-term archive grows to four immutable records while the active
cache remains bounded at two slots.

The controller and event encoder are frozen. The active cache uses
`EpisodicBindingRouter` v3, copy-on-write verifier-gated replacement, and the
generic `ExternalCapabilityEvictionPolicy`. The archive stores only learned
context/signature keys and scalar reliability/age telemetry. A stable scalar
prefix protects the mastered resident; recently admitted records remain
replaceable until they earn protection.

## Replicated result

Seeds 17 and 18 both pass:

- six replacements, with zero protected-slot evictions or avoidable
  replacements;
- twelve active-binding no-op probes, showing interleaved known traffic does
  not thrash the cache;
- four archive records, including two inactive records found again by fresh
  signatures without replaying their old training stream;
- verifier-gated retention for every replacement and successful router/archive
  reload;
- held-out victim-policy accuracy `0.8047` and `0.8184`; reward-shuffled
  controls `0.3086` and `0.3301`;
- frozen controller and event encoder, with zero replayed training examples.

Each seed used 10,231 unique verifier bits/lifetimes, 6,000 policy updates,
1,000 router updates, 1,152 retention-probe bits, and 0 replayed examples.
The reports contain the complete accounting and gate results:

- `report_seed17.json`
- `report_seed18.json`
- `sample_efficiency_ledger.json`

Run a seed with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_interleaved_binding_archive \
  --seed 17 --report-out /tmp/interleaved-archive.json
```

## Claim boundary

This is promoted as growable external episodic storage plus anti-thrashing
active-cache lifecycle behavior. It is not unrestricted memory growth,
arbitrary new computation, or general continual learning. The next pressure
test is archive scale/compression and retrieval under more than four
interleaved bindings, with reversal and corruption controls.
