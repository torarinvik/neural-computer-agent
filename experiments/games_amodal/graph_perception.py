"""Graph perception: BFS reductions into the amodal slot state (F241).

The only substrate-specific code the amodal stack needs: a vocabulary
of graph-generic reductions mapping (observation, structure) to the
same [B, SLOTS] slot state the grid stack uses, VALUES=8, ABSENT=8.

Slots:
  0  BFS distance agent -> nearest food     (metric)
  1  first-hop port toward that food        (control-relevant)
  2  BFS distance agent -> hazard           (metric)
  3  first-hop port toward the hazard
  4  agent out-degree distinct successors   (local structure)
  5  BFS distance to the SECOND food (ABSENT if none)
  6  constant 0 -- the ORIGIN slot, so "reach" is expressible in the
     unchanged pair grammar as |state - 0|
  7  constant 0

Everything downstream (plant, per-slot search, banks, goal grammar,
robust selection) is reused unchanged from the grid stack.
"""

from __future__ import annotations

import torch

from experiments.games_amodal.graph_world import NODES, PORTS

SLOTS, VALUES = 8, 8
ABSENT = VALUES


def bfs_all(edges_row):
    """[NODES, NODES] pairwise BFS distances for one row's digraph."""
    dist = torch.full((NODES, NODES), 99, dtype=torch.long)
    for s in range(NODES):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for n in frontier:
                for p in range(PORTS):
                    m = int(edges_row[n, p])
                    if dist[s, m] > d:
                        dist[s, m] = d
                        nxt.append(m)
            frontier = nxt
    return dist


def first_hop(edges_row, dist, src, dst):
    """The lowest port at src that starts a shortest path to dst."""
    if src == dst:
        return 0
    best = int(dist[src, dst])
    for p in range(PORTS):
        m = int(edges_row[src, p])
        if int(dist[m, dst]) == best - 1:
            return p
    return 0


def encode(obs, edges):
    """[B, 3, NODES] + [B, NODES, PORTS] -> [B, SLOTS] slot state."""
    B = obs.shape[0]
    out = torch.full((B, SLOTS), ABSENT, dtype=torch.long)
    out[:, 6] = 0
    out[:, 7] = 0
    for i in range(B):
        if float(obs[i, 0].max()) <= 0:
            continue
        agent = int(obs[i, 0].argmax())
        dist = bfs_all(edges[i])
        foods = [int(n) for n in (obs[i, 1] > 0).nonzero().flatten()]
        if foods:
            foods.sort(key=lambda n: int(dist[agent, n]))
            n0 = foods[0]
            out[i, 0] = min(int(dist[agent, n0]), VALUES - 1)
            out[i, 1] = first_hop(edges[i], dist, agent, n0)
            if len(foods) > 1:
                out[i, 5] = min(int(dist[agent, foods[1]]), VALUES - 1)
        hz = (obs[i, 2] > 0).nonzero().flatten()
        if hz.numel():
            h = int(hz[0])
            out[i, 2] = min(int(dist[agent, h]), VALUES - 1)
            out[i, 3] = first_hop(edges[i], dist, agent, h)
        deg = len({int(edges[i, agent, p]) for p in range(PORTS)})
        out[i, 4] = min(deg, VALUES - 1)
    return out
