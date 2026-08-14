"""Graph substrate: the amodality test world (F241).

A batched verifier with the same step/observation contract as the
grid family, on a substrate with NO geometry: 8 nodes, each with 4
outgoing ports (a random 4-regular-ish digraph per row), an agent
walking edges, food items at nodes (+1, respawn elsewhere), and a
hazard walker stepping along its own random edges (-1 and death on
co-location). Node identity is nominal; the only structure is the
edge relation.

Observation is the raw per-row state: [B, 3, NODES] one-hot planes
(agent, food, hazard) PLUS the adjacency tensor exposed separately
via `structure()` -- the substrate's percepts are relational, not
pictorial. Perception for the amodal stack lives in
graph_perception.py (BFS distances and first-hop ports), and
everything downstream of the slot state is reused unchanged.
"""

from __future__ import annotations

import torch

from experiments.games_amodal.environments import GameStep

NODES = 8
PORTS = 4


class GraphConfig:
    def __init__(self, collect: int = 0, avoid: int = 0):
        if collect < 0 or avoid < 0:
            raise ValueError("component levels cannot be negative")
        if collect + avoid == 0:
            raise ValueError("a graph world needs at least one component")
        self.collect, self.avoid = collect, avoid


class GraphVerifier:
    action_count = PORTS

    def __init__(self, config: GraphConfig, *, batch_size: int,
                 seed: int = 0):
        self.config = config
        self.batch_size = int(batch_size)
        self._gen = torch.Generator().manual_seed(seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._gen.manual_seed(int(seed))
        B = self.batch_size
        # random digraph: each node's 4 ports lead to distinct other
        # nodes; connectivity is near-certain at this density
        self.edges = torch.zeros(B, NODES, PORTS, dtype=torch.long)
        for i in range(B):
            for n in range(NODES):
                perm = torch.randperm(NODES, generator=self._gen)
                targets = [int(x) for x in perm if int(x) != n][:PORTS]
                self.edges[i, n] = torch.tensor(targets)
        self.agent = torch.randint(0, NODES, (B,), generator=self._gen)
        self.alive = torch.ones(B, dtype=torch.bool)
        self.food = torch.full((B, max(1, self.config.collect)), -1,
                               dtype=torch.long)
        for k in range(self.config.collect):
            self.food[:, k] = self._free_nodes()
        self.hazard = torch.full((B,), -1, dtype=torch.long)
        if self.config.avoid:
            self.hazard = self._free_nodes()

    def _free_nodes(self) -> torch.Tensor:
        B = self.batch_size
        out = torch.randint(0, NODES, (B,), generator=self._gen)
        for _ in range(16):
            clash = out == self.agent
            if self.food.numel():
                clash = clash | (out.unsqueeze(1) == self.food).any(dim=1)
            if not bool(clash.any()):
                break
            fresh = torch.randint(0, NODES, (int(clash.sum()),),
                                  generator=self._gen)
            out = out.clone()
            out[clash] = fresh
        return out

    def observation(self) -> torch.Tensor:
        B = self.batch_size
        obs = torch.zeros(B, 3, NODES)
        rows = torch.arange(B)
        obs[rows, 0, self.agent] = 1.0
        for k in range(self.config.collect):
            keep = (self.food[:, k] >= 0) & self.alive
            obs[rows[keep], 1, self.food[keep, k]] = 1.0
        if self.config.avoid:
            keep = self.alive
            obs[rows[keep], 2, self.hazard[keep]] = 1.0
        dead = ~self.alive
        obs[dead] = 0.0
        return obs

    def structure(self) -> torch.Tensor:
        """The substrate's relational percept: [B, NODES, PORTS]."""
        return self.edges

    def step(self, actions: torch.Tensor) -> GameStep:
        B = self.batch_size
        rows = torch.arange(B)
        reward = torch.zeros(B)
        target = self.edges[rows, self.agent, actions.long()]
        target = torch.where(self.alive, target, self.agent)
        self.agent = target
        for k in range(self.config.collect):
            got = (self.food[:, k] == target) & self.alive
            reward = reward + got.float()
            if bool(got.any()):
                fresh = self._free_nodes()
                food = self.food.clone()
                food[got, k] = fresh[got]
                self.food = food
        if self.config.avoid:
            hop = torch.randint(0, PORTS, (B,), generator=self._gen)
            self.hazard = self.edges[rows, self.hazard, hop]
            hit = (self.hazard == target) & self.alive
            reward = reward - hit.float()
            self.alive = self.alive & ~hit
        return GameStep(reward=reward, alive=self.alive.clone())
