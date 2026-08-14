import torch

from experiments.games_amodal.graph_perception import ABSENT, encode
from experiments.games_amodal.graph_world import GraphConfig, GraphVerifier


def test_collect_rewards_and_respawns() -> None:
    v = GraphVerifier(GraphConfig(collect=1), batch_size=1, seed=3)
    v.reset(seed=3)
    # walk the agent onto the food via port lookup
    food = int(v.food[0, 0])
    placed = False
    for p in range(4):
        if int(v.edges[0, int(v.agent[0]), p]) == food:
            out = v.step(torch.tensor([p]))
            assert float(out.reward[0]) == 1.0
            assert int(v.food[0, 0]) != food  # respawned elsewhere
            placed = True
            break
    if not placed:  # food not adjacent: stepping earns nothing
        out = v.step(torch.tensor([0]))
        assert float(out.reward[0]) == 0.0


def test_hazard_colocation_kills() -> None:
    torch.manual_seed(0)
    died = 0
    for seed in range(30):
        v = GraphVerifier(GraphConfig(collect=1, avoid=1), batch_size=8,
                          seed=seed)
        v.reset(seed=seed)
        g = torch.Generator().manual_seed(seed)
        for _ in range(12):
            out = v.step(torch.randint(0, 4, (8,), generator=g))
        died += int((~out.alive).sum())
    assert died > 0  # hazards actually kill under random play


def test_encode_produces_metric_slots() -> None:
    v = GraphVerifier(GraphConfig(collect=2, avoid=1), batch_size=4,
                      seed=7)
    v.reset(seed=7)
    code = encode(v.observation(), v.structure())
    assert code.shape == (4, 8)
    assert bool((code[:, 6] == 0).all())  # origin slot
    assert bool((code[:, 0] < ABSENT).all())  # food distance present
    assert bool((code[:, 1] < 4).all())  # port index in range
    # distance slots respect the graph: distance 0 means co-located
    for i in range(4):
        if int(code[i, 0]) == 0:
            agent = int(v.observation()[i, 0].argmax())
            assert float(v.observation()[i, 1, agent]) > 0
