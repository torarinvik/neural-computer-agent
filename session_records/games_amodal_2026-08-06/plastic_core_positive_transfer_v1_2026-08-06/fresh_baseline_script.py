import json, sys, torch
from experiments.games_amodal.shared_controller import (
    SharedControllerAgent, evaluate_game, train_game, trainable_parameters,
)

seed = int(sys.argv[1])
torch.manual_seed(seed)
agent = SharedControllerAgent(
    event_width=64, intention_width=32, feedback_width=16, hidden=32
)
history = train_game(
    agent, "pong",
    trainable=trainable_parameters([agent.controller, *agent.game_modules("pong")]),
    updates=600, batch_size=64, steps=64, seed=seed + 50_000,
    gamma=0.95, learning_rate=1e-3, shuffle_rewards=False,
)
masteries = [e["mastery"] for e in history]
crossed = [e["update"] for e in history if e["mastery"] >= 0.5]
report = {
    "seed": seed,
    "eval": evaluate_game(agent, "pong", batch_size=64, steps=64,
        seeds=tuple(seed + 10_000 + i for i in range(8)), gamma=0.95),
    "mean_training_mastery": sum(masteries) / len(masteries),
    "updates_to_half_mastery": None if not crossed else crossed[0],
    "no_replay": all(e["replayed_examples"] == 0.0 for e in history),
}
json.dump(report, open(sys.argv[2], "w"), indent=2, sort_keys=True)
