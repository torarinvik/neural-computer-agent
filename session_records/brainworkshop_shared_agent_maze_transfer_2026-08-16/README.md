# Shared-agent maze transfer (2026-08-16)

Status: **development planning diagnostic; not promoted**.

This record adds a procedurally generated rendered maze as a downstream task
for the existing Neural Workshop agent.  The warm arm runs two Neural Workshop
lifetimes and then enters Maze on the **same**
`CanonicalBrainWorkshopAgent` object.  The shared controller, amodal event
transport, recurrent state contract, intention bus, and protocol decoder stay
unchanged.  Maze walls, coordinates, action permutations, goals, and verifier
state remain outside the controller.

The arms are matched on two distinct target mazes:

- `workshop_warm`: the same agent after the Workshop warm-up, carrying only a
  verified world-independent exploration/rebuild operator;
- `fresh`: a new copy of the same architecture with no reusable operator;
- `stale_world_model`: a source-maze factual map transplanted into the target;
- `reward_shuffled`: the same shared-agent path with reward evidence withheld.

## Result

| arm | target replicate 0 final normalized return | target replicate 1 final normalized return | stable bits |
| --- | ---: | ---: | --- |
| Workshop warm | 0.556 | 0.764 | never / 436 |
| fresh | 0.000 | 0.083 | never / never |
| stale world model | 0.000 | 0.000 | never / never |
| reward shuffled | 0.597 | 0.000 | never / never |

The warm arm has a clear development-time learning-curve advantage over the
fresh arm on both target maps, and the stale source-map arm fails after the
layout changes.  Because the fresh arm never reached the provisional stable
threshold in either replicate, no warm/fresh transfer ratio is claimed.  The
reward-shuffled control is noisy and does not support a positive claim.

Both warm runs report `same_agent_object_for_workshop_and_maze=true` and an
unchanged controller digest.  This is evidence that the task switch crosses
the intended shared boundary; it is not evidence that neural weights have
learned maze planning.  No optimizer updates occurred in this diagnostic.

The event dictionary is a discarded development probe built from valid maze
rerenders so this first audit can isolate planning.  It is explicitly not a
deployed maze oracle; promotion would require experience-based event
discovery, then a fresh comparison with the same shared agent.

## Boundary and next gate

This record establishes the first same-agent Workshop-to-Maze composition and
the need for a real planning benchmark.  It does not yet establish promotion,
neural weight transfer, cross-frontend transfer, symbol remapping, partial
observability, occlusion, dynamic obstacles, or safe persistent identity in a
maze.  The next development gate is to train or admit a measured Workshop
planning artifact, repeat this curve with more maze replicates, and require a
stable warm advantage without confident map errors or reward-shuffled leakage.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.maze_transfer \
  --output session_records/brainworkshop_shared_agent_maze_transfer_2026-08-16
```

The complete per-task report is in `maze_transfer.json`.  The sampler and
renderer are implemented in `experiments/brainworkshop_canonical/maze_environment.py`;
the shared-agent audit is in `maze_transfer.py`.
