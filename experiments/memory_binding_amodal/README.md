# Outcome-only two-slot binding

This is a deliberately narrow diagnostic for the next memory bottleneck. Each
batch row stores two opaque slot outcomes, resets recurrent state, and must
recall one selected slot. The memory backend uses two rows and an opaque
per-trajectory scope, so identical keys in different independent batch rows
cannot overwrite one another.

The audit uses fixed writes (`write_threshold=0.0`) to isolate content-key
binding and batch isolation from the separate learned-retention question. The
v18 shared event-window address path passes the three-seed narrow gate: intact
recall is `1.0` for seeds 17, 18, and 19, while clear, corruption, swapped-slot,
and swapped-scope controls remain near chance. The corrected reward-shuffled
control remains at `0.5234` and does not promote.

This qualifies only fixed-write two-slot binding and batch isolation. It does
not qualify learned skipping, utility-based eviction, persistent episodic
memory, or cross-adapter retrieval. The earlier v17 attempt remains recorded
as a rejected rung because it exposed the missing event-stable address path.

Run a short rung with:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.memory_binding_amodal.train \
  --steps 128 --batch-size 4 --seed 17 --report-out /tmp/memory-binding.json
```
