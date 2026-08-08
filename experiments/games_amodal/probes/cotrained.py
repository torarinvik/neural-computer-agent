"""Co-trained self-addressing (F47's prediction).

Every staged design in this program worked when its stages shared a
policy (F13, F18, F40) and failed when they did not (F20, F21, F41,
F47). Probe-addressing and oracle-addressing do not share one: under an
oracle the plant conditions on a constant, under a probe it conditions
on the consequences of its own probing action. So the three parts are
trained here in ONE loop under ONE objective from update zero -- probe,
fetch, execute -- never assembled from separately validated stages.

Each episode: act with a fixed test action while the bank is withheld,
read the resulting state, fetch by its sign, then play on with what was
fetched. The whole episode is scored once and every component learns
from that single signal plus the probe's own reward prediction.
"""
from __future__ import annotations

import argparse
import json

import torch

from experiments.games_amodal.fragment_bank import (
    ContextProbe,
    FragmentBank,
    mastery,
    sample_selection,
    twins_suite,
)
from experiments.games_amodal.game_family import FamilyVerifier
from experiments.games_amodal.shared_controller import (
    SHARED_SCREEN_CHANNELS,
    SharedControllerAgent,
    pad_channels,
    trainable_parameters,
)
from experiments.games_amodal.skill_externalization import (
    artifact_events,
    ignorance_loss,
)
from neural_computer import ControllerFeedback

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=69316)
parser.add_argument("--updates", type=int, default=3000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--probe-steps", type=int, default=2)
parser.add_argument("--fragments", type=int, default=6)
parser.add_argument("--per-variant", type=int, default=2)
parser.add_argument("--diversity", type=float, default=2.0)
parser.add_argument("--ignorance-every", type=int, default=3)
parser.add_argument("--ignorance", type=float, default=0.0,
                    help="push the bank-free and decoy policies toward uniform, so the bank must be NECESSARY (F48)")
parser.add_argument("--handover", type=int, default=1200)
parser.add_argument(
    "--symmetric-plant", action="store_true",
    help="roll out both contexts each update and step once on the sum, so "
         "the plant only ever receives the mixture gradient")
args = parser.parse_args()

torch.manual_seed(args.seed)
train, _ = twins_suite()
agent = SharedControllerAgent(event_width=64, intention_width=32,
                              feedback_width=16, hidden=32,
                              event_window_capacity=8, shared_drivers=True)
bank = FragmentBank(fragments=args.fragments, tokens_per_fragment=2, width=64,
                    variants=[c.name for c in train])
probe = ContextProbe(intention_width=32)
slots = torch.nn.ParameterDict({
    "pos": torch.nn.Parameter(torch.randn(args.fragments) * 2.0),
    "neg": torch.nn.Parameter(torch.randn(args.fragments) * 2.0),
})
canonical = [list(range(args.per_variant)),
             list(range(args.per_variant, 2 * args.per_variant))]
plant = list(trainable_parameters([agent.controller, *agent.game_modules(agent.games[0])]))
params = plant + [bank.tokens] + list(slots.parameters()) + list(probe.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
decoder = agent.runtime.output_bus.decoders["keypress"]
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def test_action(observation):
    batch = observation.shape[0]
    centre = observation.shape[-1] // 2
    actions = torch.zeros(batch, dtype=torch.long)
    for row in range(batch):
        for index, (dr, dc) in enumerate(DELTAS):
            if float(observation[row, 1, centre + dr, centre + dc]) > 0:
                actions[row] = index
                break
    return actions


def episode(config, seed, *, staging, sample=True, force=None, decoy=False,
            blind=False, keep_logits=False):
    """Probe, fetch, then execute -- one continuous episode."""
    verifier = FamilyVerifier(config, batch_size=args.batch_size, seed=seed)
    verifier.reset(seed=seed)
    state = agent.controller.initial_state(args.batch_size, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(args.batch_size, agent.controller.feedback_width),
        reward=torch.zeros(args.batch_size), propensity=torch.ones(args.batch_size),
        has_feedback=torch.zeros(args.batch_size))
    rewards, logps, masks, logits_trace = [], [], [], []
    alive = torch.ones(args.batch_size, dtype=torch.bool)
    probe_pred = probe_target = None
    fragments = None
    for step in range(args.steps):
        masks.append(alive.float())
        observation = pad_channels(verifier.observation(), SHARED_SCREEN_CHANNELS)
        events = [agent.runtime.encoders["screen"](observation)]
        if fragments is not None:
            events.extend(artifact_events(
                fragments.reshape(-1, fragments.shape[-1]), args.batch_size))
        output, state = agent.runtime.step_events(events, state, feedback)
        if step < args.probe_steps:
            actions = test_action(observation)
            logps.append(torch.zeros(args.batch_size))
            propensity = torch.ones(args.batch_size)
            if step == args.probe_steps - 1:
                probe_pred = probe(output.intention.payload)
        else:
            if keep_logits:
                logits_trace.append(output.decoded["keypress"])
            decision = decoder.decide_from_logits(output.decoded["keypress"], sample=sample)
            actions = decision.key_index
            logps.append(decision.propensity.clamp_min(1e-8).log())
            propensity = decision.propensity.detach()
        outcome = verifier.step(actions)
        rewards.append(outcome.reward)
        alive = outcome.alive
        feedback = ControllerFeedback(
            action=agent.feedback_encoders["keypress"](actions),
            reward=outcome.reward, propensity=propensity,
            has_feedback=torch.ones(args.batch_size))
        if step == args.probe_steps - 1:
            probe_target = outcome.reward
            with torch.no_grad():
                sign = int(probe_pred.mean() > 0)
            key = "pos" if sign else "neg"
            if force is not None:
                chosen, logp = force, torch.zeros(())
            elif staging:
                chosen, logp = canonical[sign], torch.zeros(())
            else:
                chosen, logp = sample_selection(slots[key], args.per_variant, greedy=not sample)
            fragments = bank.fetch(chosen)
            if decoy:
                noise = torch.randn_like(fragments)
                fragments = noise * (
                    fragments.detach().norm() / noise.norm().clamp_min(1e-12)
                )
            if blind:
                fragments = None
            episode.last = (key, chosen, logp, sign)
        state = state.detached() if sample else state
    reward_matrix = torch.stack(rewards, dim=1)
    mask_matrix = torch.stack(masks, dim=1)
    returns = torch.zeros_like(reward_matrix)
    running = torch.zeros(args.batch_size)
    for pos in range(reward_matrix.shape[1] - 1, -1, -1):
        running = reward_matrix[:, pos] + 0.95 * running
        returns[:, pos] = running
    advantage = returns.detach()
    advantage = advantage - (advantage * mask_matrix).sum() / mask_matrix.sum().clamp_min(1)
    return {"logits": torch.stack(logits_trace, dim=1) if logits_trace else None,
            "reward": reward_matrix, "mask": mask_matrix, "advantage": advantage,
            "logp": torch.stack(logps, dim=1), "probe_pred": probe_pred,
            "probe_target": probe_target}



def earned(out):
    """Score ONLY what happened after the probe (F53).

    The probe runs a fixed hand-coded action that steps onto the
    positive-plane item -- which IS choiceA's task. Scoring the whole
    episode credits the harness: with no agent at all, probe + frozen
    actions scores 0.961 on choiceA and 0.004 on choiceB, reproducing
    every gate the loop reported. Slicing the probe steps off restores
    symmetry (0.238 / 0.180) and makes the gates measure the agent.
    """

    return {
        "total_reward": out["reward"][:, args.probe_steps:].sum(dim=1),
        "mask": out["mask"][:, args.probe_steps:],
    }


def context_loss(config, update, staging):
    """One context's whole objective."""
    out = episode(config, args.seed + update, staging=staging)
    key, chosen, logp, _ = episode.last
    terms = out["advantage"] * out["logp"] * out["mask"]
    loss = -terms.sum() / terms.shape[0]
    loss = loss + (out["probe_pred"] - out["probe_target"].detach()).square().mean()
    score = mastery(earned(out), config)
    loss = loss - (score - baseline[config.name]) * logp
    baseline[config.name] = 0.9 * baseline[config.name] + 0.1 * score
    if staging:
        target = torch.zeros(args.fragments); target[chosen] = 1.0 / len(chosen)
        loss = loss + torch.nn.functional.kl_div(
            torch.log_softmax(slots[key], dim=-1), target, reduction="sum")
    elif args.diversity > 0:
        loss = loss + args.diversity * (
            torch.softmax(slots["pos"], -1) * torch.softmax(slots["neg"], -1)).sum()
    if args.ignorance > 0.0 and update % args.ignorance_every == 0:
        # F48: without this the agent solves the task from its own probe
        # outcome held in working memory and the fetched fragment is
        # decorative. Requiring the bank-free and decoy policies to be
        # UNINFORMATIVE makes the bank necessary rather than merely
        # able to override.
        for kind in ("blind", "decoy"):
            out2 = episode(config, args.seed + 800_000 + update, staging=staging,
                           blind=(kind == "blind"), decoy=(kind == "decoy"),
                           keep_logits=True)
            if out2["logits"] is not None:
                loss = loss + args.ignorance * ignorance_loss(
                    out2["logits"], out2["mask"][:, -out2["logits"].shape[1]:]
                )
    return loss


baseline = {c.name: 0.0 for c in train}
for update in range(args.updates):
    staging = update < args.handover
    if args.symmetric_plant:
        # Galashov et al. (ICLR 2019): what a default policy absorbs is
        # set by what it can SEE, not by how hard you penalise it. Their
        # default is denied task-identifying input, so it converges to
        # the marginal by construction.
        #
        # Our plant is the default-policy analogue. Sampling ONE context
        # per update hands it a context-specific gradient every step,
        # which is the channel that lets it specialise to one twin. Here
        # both contexts are rolled out every update and their losses
        # summed before a single step, so the plant only ever moves
        # along the mixture direction. F50 measured these two gradients
        # as genuinely conflicting, so the conflicting components
        # largely cancel and what survives is the common direction --
        # "read the fragment and do what it says". Anything
        # context-specific then has nowhere to live except the bank.
        loss = None
        for config in train:
            part = context_loss(config, update, staging)
            loss = part if loss is None else loss + part
    else:
        # F10 + F23: uniform alternation lets one twin take the plant.
        # Sample the laggard more often, with a uniform floor so the
        # leader keeps a maintenance ration. Both are promoted.
        weights = torch.tensor([-baseline[c.name] for c in train]) / 0.25
        probs = torch.softmax(weights, dim=-1)
        probs = 0.5 * probs + 0.5 / len(train)
        config = train[int(torch.multinomial(probs, 1))]
        loss = context_loss(config, update, staging)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()

report = {"seed": args.seed, "signs": {}, "selection": {}, "train": {},
          "cross_fed": {}, "decoy": {}, "withheld": {}}
for config in train:
    scores = []
    for index in range(4):
        with torch.no_grad():
            out = episode(config, args.seed + 60_000 + index, staging=False, sample=False)
        key, chosen, _, sign = episode.last
        scores.append(mastery(earned(out), config))
    report["signs"][config.name] = sign
    report["selection"][config.name] = chosen
    report["train"][config.name] = float(torch.tensor(scores).mean())

# Causal gates. Cross-feeding must INVERT behaviour (the specification
# signature); a norm-matched decoy must destroy it; and the two must
# differ, or the bank is merely a presence cue.
names = [c.name for c in train]
for index, config in enumerate(train):
    other = report["selection"][names[1 - index]]
    with torch.no_grad():
        out = episode(config, args.seed + 70_000, staging=False, sample=False, force=other)
    report["cross_fed"][f"{config.name}<-{names[1-index]}"] = mastery(
        earned(out), config)
    with torch.no_grad():
        out = episode(config, args.seed + 71_000, staging=False, sample=False,
                      force=report["selection"][config.name], decoy=True,
                      keep_logits=True)
    report["decoy"][config.name] = mastery(
        earned(out), config)
    # Mechanism diagnostic: near-uniform logits with a surviving greedy
    # argmax mean the ignorance objective flattened confidence but could
    # not flip the sign of the residual tilt toward the bank-free default.
    #
    # Masked by `alive`. The unmasked version was wrong and nearly cost a
    # false conclusion: the episode keeps stepping after death, those
    # steps drift to uniform, and averaging over them reported entropy
    # 1.3859 against ln(4)=1.38629 -- "the policy is exactly uniform" --
    # for a policy that scored 0.875 when sampled, which is impossible
    # when measured chance is 0.371. F46's rule, third instance: measure
    # the signal over the steps that actually determined behaviour.
    log_probs = torch.log_softmax(out["logits"], dim=-1)
    live = out["mask"][:, -out["logits"].shape[1]:]
    weight = live.sum().clamp_min(1.0)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    report.setdefault("decoy_entropy", {})[config.name] = float(
        (entropy * live).sum() / weight)
    report.setdefault("decoy_max_prob", {})[config.name] = float(
        (log_probs.exp().max(dim=-1).values * live).sum() / weight)
    report.setdefault("decoy_live_fraction", {})[config.name] = float(
        live.mean())
    report.setdefault("decoy_entropy_unmasked", {})[config.name] = float(
        entropy.mean())
    with torch.no_grad():
        out = episode(config, args.seed + 72_000, staging=False, sample=True,
                      force=report["selection"][config.name], decoy=True)
    report.setdefault("decoy_sampled", {})[config.name] = mastery(
        earned(out), config)
print(json.dumps(report))
