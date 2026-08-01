"""Tiny reward-only learning probe on the Brain Workshop-style gym.

This is deliberately the first learning rung rather than a claim of mastery:
vision-only n-back-1, two opaque keypress actions, scalar verifier reward, and
an external encoder/controller/output-decoder decomposition.  The verifier
keeps the match target private; it is used only by ``score_action``.

The default run is short enough to act as a gate.  A healthy run should show
live gradients and a changing policy before it is given a longer budget.  The
script also evaluates a history-reset control, which prevents a lucky
per-frame classifier from being mistaken for working memory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .amodal_interface import AmodalEvent
from .amodal_runtime import AmodalInputBus, OpaqueProtocolDecoder
from .brainworkshop_gym import (
    BrainWorkshopAudioEncoder, BrainWorkshopConfig,
    BrainWorkshopVisionEncoder, generate_brainworkshop_episode)
from .model import UnifiedCognitiveController


@dataclass
class Rollout:
    log_probs: torch.Tensor
    entropies: torch.Tensor
    rewards: torch.Tensor
    actions: torch.Tensor
    values: torch.Tensor


class BrainWorkshopPolicy(nn.Module):
    """N encoders -> amodal bus -> one controller -> one decoder."""

    def __init__(self, *, width: int = 32, intention_width: int = 16,
                 workspace_slots: int = 2, relation_adapter_width: int = 0,
                 event_memory_adapter_width: int = 0,
                 retrieved_memory_adapter_width: int = 0,
                 input_bus_residual_width: int = 0,
                 external_memory_adapter_width: int = 0,
                 modality: str = "vision",
                 modalities: tuple[str, ...] | None = None,
                 controller: UnifiedCognitiveController | None = None) -> None:
        super().__init__()
        if modalities is None:
            modalities = (modality,)
        modalities = tuple(modalities)
        if not modalities or any(value not in ("vision", "audio")
                                 for value in modalities):
            raise ValueError("modalities must contain vision and/or audio")
        if len(set(modalities)) != len(modalities):
            raise ValueError("modalities must be unique")
        self.modalities = modalities
        # Keep the old attribute for single-stream callers and reports.
        self.modality = modalities[0] if len(modalities) == 1 else "multi"
        self.action_bits = tuple(
            1 if name == "vision" else 2 for name in modalities)

        self.controller = controller or UnifiedCognitiveController(
            width=width, workspace_slots=workspace_slots,
            intention_width=intention_width,
            relation_adapter_width=relation_adapter_width,
            relation_adapter_layer_norm=True,
            relation_adapter_gated=True,
            event_memory_adapter_width=event_memory_adapter_width,
            retrieved_memory_adapter_width=retrieved_memory_adapter_width)
        width = self.controller.width
        intention_width = self.controller.intention_width
        self.encoders = nn.ModuleDict({
            name: (
                BrainWorkshopVisionEncoder(width)
                if name == "vision" else BrainWorkshopAudioEncoder(width)
            )
            for name in modalities
        })
        self.input_bus = AmodalInputBus(
            width, residual_hidden=input_bus_residual_width)
        self.external_memory_adapter = None
        if external_memory_adapter_width:
            self.external_memory_adapter = nn.Sequential(
                nn.Linear(width * 3, external_memory_adapter_width),
                nn.GELU(),
                nn.Linear(external_memory_adapter_width, width),
            )
            # The adapter starts as an exact no-op. It can only earn a change
            # to the RAM snapshot through the verifier's scalar outcome.
            nn.init.zeros_(self.external_memory_adapter[-1].weight)
            nn.init.zeros_(self.external_memory_adapter[-1].bias)
        # The legacy model owns adapters for compatibility.  The probe removes
        # them so only this independent output adapter formats the intention.
        self.controller.vision = None
        self.controller.actuator = None
        self.decoder = OpaqueProtocolDecoder(
            intention_width, commands=1 << len(modalities))
        # Optional reward-only critic.  It predicts return from the recurrent
        # state; it never sees verifier targets or modality/task metadata.
        self.value_head = nn.Linear(self.controller.width, 1)

    @property
    def encoder(self) -> nn.Module:
        """Legacy single-stream encoder accessor."""
        if len(self.modalities) != 1:
            raise AttributeError("multi-stream policies have .encoders")
        return self.encoders[self.modalities[0]]

    def action_mask(self, action: int) -> int:
        """Translate decoder class 0..2**N-1 to verifier bitmask."""
        if action < 0 or action >= (1 << len(self.modalities)):
            raise ValueError("decoder action is outside the modality mask")
        return sum(
            bit for index, bit in enumerate(self.action_bits)
            if action & (1 << index))

    def initial_state(self, batch: int, device: torch.device):
        return self.controller.initial_state(batch, device=device)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def _make_batch(config: BrainWorkshopConfig, *, batch_size: int,
                seed: int, device: torch.device,
                modalities: tuple[str, ...]):
    episodes = [generate_brainworkshop_episode(
        config, seed=seed + index, device=device) for index in range(batch_size)]
    observations = {
        modality: torch.stack([
            torch.stack([
                observation.vision if modality == "vision"
                else observation.audio
                for observation in episode.observations])
            for episode in episodes])
        for modality in modalities
    }
    return episodes, observations


def _event_payloads(
        policy: BrainWorkshopPolicy, observations: dict[str, torch.Tensor],
        episodes, trial: int, device: torch.device, *, oracle_events: bool,
        stimulus_indices: torch.Tensor | None = None) -> list[AmodalEvent]:
    """Encode every declared modality into one opaque event for this trial."""
    events = []
    indices = (
        [trial] * len(episodes) if stimulus_indices is None
        else [int(value) for value in stimulus_indices])
    for modality in policy.modalities:
        if oracle_events:
            # Diagnostic-only localization control. It bypasses pixels or
            # waveforms with a verifier-side one-hot event; its result is never
            # promoted as a learned representation.
            values = torch.tensor(
                [(episode.stimuli[index].position
                  if modality == "vision" else episode.stimuli[index].audio)
                 for episode, index in zip(episodes, indices)],
                dtype=torch.long, device=device)
            payload = F.one_hot(values, num_classes=8).to(torch.float32)
            if payload.shape[1] < policy.controller.width:
                payload = F.pad(
                    payload, (0, policy.controller.width - payload.shape[1]))
            payload = payload[:, :policy.controller.width]
        else:
            payload = policy.encoders[modality](observations[modality][:, trial])
        events.append(AmodalEvent(payload=payload))
    return events


def _rollout(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig,
             *, batch_size: int, seed: int, device: torch.device,
             sample: bool, reset_history: bool = False,
             shuffle_time: bool = False,
             oracle_events: bool = False,
             external_history: bool = False) -> Rollout:
    episodes, observations = _make_batch(
        config, batch_size=batch_size, seed=seed, device=device,
        modalities=policy.modalities)
    stimulus_indices = None
    if shuffle_time:
        # This is an adversarial control: preserve the frame multiset but break
        # the temporal relation that n-back requires.
        permutations = torch.stack([
            torch.randperm(config.trials, device=device)
            for _ in range(batch_size)
        ])
        observations = {
            modality: torch.stack([
                values[index, permutations[index]]
                for index in range(batch_size)])
            for modality, values in observations.items()
        }
        stimulus_indices = permutations
    state = policy.initial_state(batch_size, device)
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    has_feedback = torch.zeros(batch_size, device=device)
    previous_event = None
    log_probs, entropies, rewards, actions, values = [], [], [], [], []
    for trial in range(config.trials):
        if reset_history and trial:
            state = policy.initial_state(batch_size, device)
            # Do not mutate tensors already retained in the rollout log.
            # In-place zeroing here would silently erase earlier rewards and
            # actions, making the history-reset control look better or worse
            # for purely bookkeeping reasons.
            previous_action = torch.zeros(
                batch_size, dtype=torch.long, device=device)
            previous_reward = torch.zeros(batch_size, device=device)
            has_feedback = torch.zeros(batch_size, device=device)
            previous_event = None
        events = _event_payloads(
            policy, observations, episodes, trial, device,
            oracle_events=oracle_events,
            stimulus_indices=(
                stimulus_indices[:, trial] if stimulus_indices is not None
                else None))
        event = policy.input_bus(events)
        retrieved_memory = (
            previous_event if external_history and previous_event is not None
            else torch.zeros_like(event.payload))
        if (external_history and previous_event is not None
                and policy.external_memory_adapter is not None):
            memory_features = torch.cat([
                event.payload, retrieved_memory,
                event.payload * retrieved_memory], dim=-1)
            retrieved_memory = retrieved_memory + (
                policy.external_memory_adapter(memory_features))
        core, state = policy.controller.step_event(
            event, state, previous_action, previous_reward, has_feedback,
            retrieved_memory=retrieved_memory)
        values.append(policy.value_head(state.hidden).squeeze(-1))
        logits = policy.decoder(core.intent_event)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(dim=-1)
        reward = torch.tensor([
            episode.score_action(
                trial, policy.action_mask(int(action[index])), latency_ms=0.0)
            for index, episode in enumerate(episodes)
        ], device=device)
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        rewards.append(reward)
        actions.append(action)
        previous_action = action
        previous_reward = reward
        has_feedback.fill_(1.0)
        previous_event = event.payload
    return Rollout(
        torch.stack(log_probs), torch.stack(entropies),
        torch.stack(rewards), torch.stack(actions), torch.stack(values))


def _supervised_step(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig,
                     *, batch_size: int, seed: int,
                     device: torch.device,
                     oracle_events: bool = False,
                     external_history: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Disposable verifier-label ceiling probe.

    Labels are used only to answer the architecture question and are never
    part of the reward-only claim.  Feedback still comes from the sampled
    policy action's scalar verifier reward, so this checks the same recurrent
    interface rather than feeding the answer back as an input.
    """
    episodes, observations = _make_batch(
        config, batch_size=batch_size, seed=seed, device=device,
        modalities=policy.modalities)
    state = policy.initial_state(batch_size, device)
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    has_feedback = torch.zeros(batch_size, device=device)
    previous_event = None
    losses, rewards = [], []
    for trial in range(config.trials):
        events = _event_payloads(
            policy, observations, episodes, trial, device,
            oracle_events=oracle_events)
        event = policy.input_bus(events)
        retrieved_memory = (
            previous_event if external_history and previous_event is not None
            else torch.zeros_like(event.payload))
        if (external_history and previous_event is not None
                and policy.external_memory_adapter is not None):
            memory_features = torch.cat([
                event.payload, retrieved_memory,
                event.payload * retrieved_memory], dim=-1)
            retrieved_memory = retrieved_memory + (
                policy.external_memory_adapter(memory_features))
        core, state = policy.controller.step_event(
            event, state, previous_action, previous_reward, has_feedback,
            retrieved_memory=retrieved_memory)
        logits = policy.decoder(core.intent_event)
        targets = torch.tensor(
            [sum(
                (1 << index) for index, bit in enumerate(policy.action_bits)
                if episode.verifier_targets()[trial] & bit)
             for episode in episodes],
            dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, targets))
        action = logits.argmax(dim=-1)
        reward = torch.tensor([
            episode.score_action(
                trial, policy.action_mask(int(action[index])), latency_ms=0.0)
            for index, episode in enumerate(episodes)
        ], device=device)
        rewards.append(reward)
        previous_action = action
        previous_reward = reward
        has_feedback = torch.ones(batch_size, device=device)
        previous_event = event.payload
    return torch.stack(losses).mean(), torch.stack(rewards)


@torch.no_grad()
def _evaluate(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig, *,
              count: int, seed: int, device: torch.device,
              reset_history: bool = False, shuffle_time: bool = False,
              oracle_events: bool = False,
              external_history: bool = False) -> dict:
    rollout = _rollout(
        policy, config, batch_size=count, seed=seed, device=device,
        sample=False, reset_history=reset_history, shuffle_time=shuffle_time,
        oracle_events=oracle_events, external_history=external_history)
    rewards = rollout.rewards
    return {
        "accuracy": float((rewards > 0).float().mean()),
        "mean_reward": float(rewards.mean()),
        "trial_count": int(rewards.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-in", type=Path,
                        help="optional promoted controller to keep frozen")
    parser.add_argument("--supervised-diagnostic", action="store_true",
                        help="use disposable verifier labels for a ceiling probe")
    parser.add_argument("--encoder-in", type=Path,
                        help="optional self-supervised encoder checkpoint")
    parser.add_argument("--vision-encoder-in", type=Path,
                        help="vision encoder checkpoint for a multi-stream run")
    parser.add_argument("--audio-encoder-in", type=Path,
                        help="audio encoder checkpoint for a multi-stream run")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="keep a loaded encoder fixed during the probe")
    parser.add_argument("--value-baseline", action="store_true",
                        help="train a reward-only recurrent value baseline")
    parser.add_argument("--oracle-events", action="store_true",
                        help="diagnostic-only verifier-side one-hot events")
    parser.add_argument("--external-history", action="store_true",
                        help="diagnostic generic one-step RAM snapshot")
    parser.add_argument("--relation-adapter-width", type=int, default=0,
                        choices=(0, 32, 64))
    parser.add_argument("--event-memory-adapter-width", type=int, default=0,
                        choices=(0, 32, 64))
    parser.add_argument("--retrieved-memory-adapter-width", type=int, default=0,
                        choices=(0, 32, 64))
    parser.add_argument("--input-bus-residual-width", type=int, default=0,
                        choices=(0, 32, 64))
    parser.add_argument("--external-memory-adapter-width", type=int, default=0,
                        choices=(0, 32, 64))
    parser.add_argument("--seed", type=int, default=44011)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--position-vocab", type=int, default=2,
                        choices=(2, 4, 8))
    parser.add_argument("--modality", choices=("vision", "audio"),
                        default="vision")
    parser.add_argument(
        "--modalities", type=str, default="",
        help="comma-separated modality list, e.g. vision,audio")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 4 or args.eval_count < 4:
        raise ValueError("updates, batch size, and eval count must be positive")
    if not 0.0 < args.discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    modalities = tuple(
        item.strip() for item in args.modalities.split(",") if item.strip())
    if not modalities:
        modalities = (args.modality,)
    if not modalities or any(item not in ("vision", "audio")
                             for item in modalities):
        raise ValueError("--modalities must contain vision and/or audio")
    if len(set(modalities)) != len(modalities):
        raise ValueError("--modalities must not contain duplicates")
    config = BrainWorkshopConfig(
        n_back=1, trials=args.trials, position_vocab=args.position_vocab,
        modalities=modalities, trial_ms=1_000,
        # Independent seeded match flags prevent a fixed-count clock policy
        # from looking like working memory.  The balanced mode remains the
        # default gym control; this training rung explicitly removes that
        # shortcut while keeping all outcomes verifier-private.
        balanced_matches=False,
    )
    frozen_controller = None
    checkpoint_payload = None
    if args.checkpoint_in:
        checkpoint_payload = torch.load(
            args.checkpoint_in, map_location=device, weights_only=False)
        controller_configuration = dict(
            checkpoint_payload["model_configuration"])
        # Policy checkpoints contain the controller under the ``controller.``
        # prefix. Keep loading compatible with older controller-only artifacts
        # while making the frozen-controller ablation actually executable.
        controller_configuration.setdefault(
            "relation_adapter_width", args.relation_adapter_width)
        controller_configuration.setdefault(
            "event_memory_adapter_width", args.event_memory_adapter_width)
        controller_configuration.setdefault(
            "retrieved_memory_adapter_width",
            args.retrieved_memory_adapter_width)
        frozen_controller = UnifiedCognitiveController(
            **controller_configuration).to(device)
        # The policy checkpoint intentionally strips legacy modality/action
        # branches before serialization. Mirror that shape before loading the
        # controller-only subset.
        frozen_controller.vision = None
        frozen_controller.actuator = None
        state_dict = checkpoint_payload["state_dict"]
        controller_state = {
            key.removeprefix("controller."): value
            for key, value in state_dict.items()
            if key.startswith("controller.")
        }
        frozen_controller.load_state_dict(
            controller_state if controller_state else state_dict)
    policy = BrainWorkshopPolicy(
        relation_adapter_width=args.relation_adapter_width,
        event_memory_adapter_width=args.event_memory_adapter_width,
        retrieved_memory_adapter_width=args.retrieved_memory_adapter_width,
        input_bus_residual_width=args.input_bus_residual_width,
        # This is an external RAM-side adapter, not a controller weight.
        external_memory_adapter_width=args.external_memory_adapter_width,
        modalities=modalities,
        controller=frozen_controller).to(device)
    if checkpoint_payload is not None:
        # Reuse compatible non-controller pieces when continuing a policy
        # checkpoint. This makes the frozen-controller test a continuation
        # experiment rather than a random decoder reinitialization, while
        # shape-mismatched dual-stream pieces are intentionally skipped.
        current_state = policy.state_dict()
        compatible = {
            key: value for key, value in checkpoint_payload["state_dict"].items()
            if key in current_state and current_state[key].shape == value.shape
        }
        policy.load_state_dict(compatible, strict=False)
    encoder_paths = {}
    if args.encoder_in:
        if len(modalities) != 1:
            raise ValueError("--encoder-in is for a single-stream run")
        encoder_paths[modalities[0]] = args.encoder_in
    if args.vision_encoder_in:
        encoder_paths["vision"] = args.vision_encoder_in
    if args.audio_encoder_in:
        encoder_paths["audio"] = args.audio_encoder_in
    for encoder_modality, encoder_path in encoder_paths.items():
        if encoder_modality not in modalities:
            raise ValueError(
                f"{encoder_modality} encoder is not in --modalities")
        encoder_payload = torch.load(
            encoder_path, map_location=device, weights_only=False)
        if encoder_payload.get("event_width") != policy.controller.width:
            raise ValueError("encoder width does not match controller width")
        encoder_modalities = tuple(
            encoder_payload.get("config", {}).get("modalities", ()))
        if encoder_modalities and encoder_modalities != (encoder_modality,):
            raise ValueError("encoder modality does not match the policy")
        policy.encoders[encoder_modality].load_state_dict(
            encoder_payload["encoder_state_dict"])
    if args.freeze_encoder and set(encoder_paths) != set(modalities):
        raise ValueError(
            "--freeze-encoder requires one checkpoint per declared modality")
    if args.freeze_encoder:
        for encoder in policy.encoders.values():
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
    if frozen_controller is not None:
        for parameter in policy.controller.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    started = time.perf_counter()
    history = []
    before = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 10_000,
        device=device, oracle_events=args.oracle_events,
        external_history=args.external_history)
    for update in range(1, args.updates + 1):
        policy.train()
        if args.supervised_diagnostic:
            loss, diagnostic_rewards = _supervised_step(
                policy, config, batch_size=args.batch_size,
                seed=args.seed + update * 1_000, device=device,
                oracle_events=args.oracle_events,
                external_history=args.external_history)
            batch_accuracy = float((diagnostic_rewards > 0).float().mean())
            batch_mean_reward = float(diagnostic_rewards.mean())
            policy_entropy = 0.0
            value_loss = torch.zeros((), device=device)
        else:
            rollout = _rollout(
                policy, config, batch_size=args.batch_size,
                seed=args.seed + update * 1_000, device=device, sample=True,
                oracle_events=args.oracle_events,
                external_history=args.external_history)
            returns = torch.zeros_like(rollout.rewards)
            running = torch.zeros(args.batch_size, device=device)
            for trial in range(args.trials - 1, -1, -1):
                running = rollout.rewards[trial] + args.discount * running
                returns[trial] = running
            # A per-time baseline uses only verifier rewards from this batch;
            # it is a variance reducer, not an additional task label.
            if args.value_baseline:
                advantages = returns - rollout.values
                value_loss = F.smooth_l1_loss(
                    rollout.values, returns.detach())
                loss = -(
                    advantages.detach() * rollout.log_probs).mean()
                loss = loss + 0.5 * value_loss
            else:
                advantages = returns - returns.mean(dim=1, keepdim=True)
                value_loss = torch.zeros((), device=device)
                loss = -(
                    advantages.detach() * rollout.log_probs).mean()
            loss = loss - args.entropy_coef * rollout.entropies.mean()
            batch_accuracy = float((rollout.rewards > 0).float().mean())
            batch_mean_reward = float(rollout.rewards.mean())
            policy_entropy = float(rollout.entropies.mean().detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.parameters(), 1.0).detach())
        optimizer.step()
        record = {
            "update": update,
            "loss": float(loss.detach()),
            "batch_accuracy": batch_accuracy,
            "batch_mean_reward": batch_mean_reward,
            "policy_entropy": policy_entropy,
            "value_loss": float(value_loss.detach()),
            "gradient_norm": gradient_norm,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    policy.eval()
    after = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 10_000,
        device=device, oracle_events=args.oracle_events,
        external_history=args.external_history)
    reset = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 20_000,
        device=device, reset_history=True, oracle_events=args.oracle_events,
        external_history=args.external_history)
    shuffled = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 30_000,
        device=device, shuffle_time=True, oracle_events=args.oracle_events,
        external_history=args.external_history)
    report = {
        "experiment": "brainworkshop_multimodal_nback1_reward_only",
        "objective": (
            "supervised_diagnostic" if args.supervised_diagnostic
            else "reward_only_value_baseline" if args.value_baseline
            else "reward_only"),
        "encoder_checkpoints": {
            name: str(path) for name, path in encoder_paths.items()},
        "encoder_frozen": bool(args.freeze_encoder),
        "controller_frozen": frozen_controller is not None,
        "oracle_events": bool(args.oracle_events),
        "external_history": bool(args.external_history),
        "relation_adapter_width": args.relation_adapter_width,
        "event_memory_adapter_width": args.event_memory_adapter_width,
        "retrieved_memory_adapter_width": args.retrieved_memory_adapter_width,
        "external_memory_adapter_width": args.external_memory_adapter_width,
        "device": str(device),
        "config": config.__dict__,
        "modality": policy.modality,
        "modalities": modalities,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "unique_training_lifetimes": args.updates * args.batch_size,
        "before": before,
        "after": after,
        "history_reset_control": reset,
        "time_shuffle_control": shuffled,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gate": {
            "gradient_alive": all(
                entry["gradient_norm"] > 1e-8 for entry in history),
            "policy_changed": after["mean_reward"] != before["mean_reward"],
            "reset_control_recorded": True,
            "accepted_for_longer_run": (
                after["accuracy"] > before["accuracy"] + 0.05
                and after["accuracy"] > 0.60
                and reset["accuracy"] < after["accuracy"] - 0.05
                and shuffled["accuracy"] < after["accuracy"] - 0.05),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "brainworkshop-policy.v1",
            "model_configuration": {
                "width": policy.controller.width,
                "intention_width": policy.controller.intention_width,
                "workspace_slots": policy.controller.workspace_slots,
                "relation_adapter_width": args.relation_adapter_width,
                "event_memory_adapter_width": args.event_memory_adapter_width,
                "retrieved_memory_adapter_width": (
                    args.retrieved_memory_adapter_width),
            },
            "state_dict": policy.state_dict(),
            "config": config.__dict__,
            "report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "final": report["gate"], "before": before, "after": after,
        "history_reset": reset, "time_shuffle": shuffled,
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
