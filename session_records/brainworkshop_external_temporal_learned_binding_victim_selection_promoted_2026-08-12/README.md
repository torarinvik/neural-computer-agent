# Learned episodic binding victim selection promotion

This two-seed audit adds autonomous replacement choice to the bounded online
binding lifecycle. A generic external eviction policy receives only an
incoming opaque episode signature and generic reliability/age telemetry for
candidate slots. It learns from one scalar utility per attempted candidate,
then selects the victim without physical slot indices, semantic names, or a
correct-row label.

In the live capacity-two router, slot A is protected and slot B is weak. The
learned policy selected physical slot B under both forward and reversed
candidate order. The verifier accepted both copy-on-write transactions only
after sibling A and new binding C passed held-out retention probes.

Across seeds 17 and 18:

- held-out generic victim transfer was `0.8047` and `0.8184`;
- reward-shuffled controls were `0.3086` and `0.3301`;
- both candidate orders selected the weak physical slot;
- both transactions retained A and acquired C;
- the controller and learned event encoder received zero updates.

This promotes bounded learned maintenance choice, not universal eviction
economics, unrestricted memory growth, or general continual learning.

## Accounting

Each seed used `8,792` unique verifier bits/lifetimes: 3,000 active policy
utilities, 3,000 shuffled-policy utilities, 1,000 router utilities, 768
retention-probe outcomes, and 1,024 held-out diagnostics. No examples were
replayed for training.

## Evidence

- `report_seed17.json`
- `report_seed18.json`
- schema: `neural-computer.brainworkshop-external-temporal-learned-binding-victim-selection.v1`
- policy ABI: `neural-computer.external-capability-eviction-policy.v1`
