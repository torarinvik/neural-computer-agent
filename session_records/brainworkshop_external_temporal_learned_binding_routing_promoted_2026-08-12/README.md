# Learned episodic binding routing promotion

This two-seed audit promotes a bounded external binding-discovery primitive.
Two anonymous learned-event trajectory families provision two opaque memory
slots from their first observed context snapshots. An external episodic
context encoder then learns to route fresh trajectories using only the scalar
utility of the slot that was actually attempted. The canonical controller and
learned event encoder remain frozen, and no examples are replayed.

The route reached `1.0000` balanced accuracy on both seeds. Reversing the
physical candidate order also reached `1.0000` on both seeds. Freezing the
router after training and exact state reload preserved the route exactly. A
reward-shuffled control stayed at `0.5000` balanced accuracy on both seeds.

This does not establish autonomous ontology formation, unrestricted slot
growth, or general continual learning. Slot provisioning is still bounded and
the initial opaque key snapshots are supplied by the external memory manager.
The next pressure is online discovery and verified replacement of more than
two bindings under capacity pressure, with no semantic family index.

## Accounting

Each promoted seed used `1,000` fresh attempted route utilities, `1,000`
router optimizer updates, zero replayed examples, and zero controller
updates. The reward-shuffled control used a separate `1,000` fresh utilities.

## Evidence

- `report_seed17.json`
- `report_seed18.json`
- schema: `neural-computer.brainworkshop-external-temporal-learned-binding-routing.v1`
- router ABI: `neural-computer.episodic-binding-router.v1`
