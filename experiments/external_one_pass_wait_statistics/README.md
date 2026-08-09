# Replay-free wait statistics

This audit tests the frozen-controller transport boundary, not a new
controller head. `EventWaitStatistics` consumes generic timestamp-buffer
features and scalar wait utility once into bounded ridge sufficient statistics.
It retains no feature rows, optimizer state, or replay buffer.

Run it with:

```bash
PYTHONPATH=src:. .venv/bin/python experiments/external_one_pass_wait_statistics/train.py \
  --seed 2301 --report-out /tmp/external-one-pass-wait-2301.json
```

The promoted rung requires replicated learned waiting for a delayed partner and
learned release after an absent partner, persistence, retention of the learned
delay rule after new observations, zero replay, and a frozen controller. It is
bounded age/coverage learning, not general temporal inference or unrestricted
continual learning.
