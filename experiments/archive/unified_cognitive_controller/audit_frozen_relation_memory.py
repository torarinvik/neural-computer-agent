"""Audit frozen-weight acquisition of a context-conditioned relation rule.

The visual relation controller is frozen. A generic episodic store receives
only controller-produced event keys, the agent's attempted opaque action, and
the scalar verifier outcome. Successful action intentions are retained as
memory values and decoded by the already-frozen output adapter. No task label,
correct unattempted action, or context ID enters the memory path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController


def _digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _event_output(
        model: UnifiedCognitiveController,
        frames: torch.Tensor,
        device: torch.device) -> object:
    count = frames.shape[0]
    action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output, _ = model.step(
        frames, model.initial_state(count, device=device),
        action, zeros, zeros)
    return output


def _frames_and_batches(
        seed: int, count: int, device: torch.device):
    """Make paired relation renders with a public, two-level context marker."""
    normal = generate_lifetimes(
        count, 3, seed=seed, heldout=True, task="pair_relation",
        appearance="bars", support_trials=1, device=device)
    reversed_relation = generate_lifetimes(
        count, 3, seed=seed, heldout=True, task="pair_relation",
        reverse_contexts=True, appearance="bars", support_trials=1,
        device=device)
    generator = torch.Generator().manual_seed(seed + 900_000)
    balanced = torch.arange(count) % 2
    context = balanced[torch.randperm(count, generator=generator)].to(device)
    levels = torch.where(
        context == 0,
        torch.as_tensor(0.15, device=device),
        torch.as_tensor(0.85, device=device),
    )
    # This corner is clear of both relation objects. It is sensory input, not
    # verifier metadata; only these pixels reach the frozen controller.
    marker = levels[:, None, None, None, None].expand(
        count, 3, 3, 3, 8)
    normal_frames = normal.frames.clone()
    reversed_frames = reversed_relation.frames.clone()
    normal_frames[:, :, :, :3, :8] = marker
    reversed_frames[:, :, :, :3, :8] = marker
    return normal_frames, reversed_frames, normal, reversed_relation, context


@torch.no_grad()
def _memory_for_mapping(
        model: UnifiedCognitiveController,
        normal_frames: torch.Tensor,
        reversed_frames: torch.Tensor,
        normal_batch,
        reversed_batch,
        context: torch.Tensor,
        remap_flip: int,
        device: torch.device) -> tuple[DiskLatentMemory, torch.Tensor, torch.Tensor]:
    normal_output = _event_output(model, normal_frames[:, 0], device)
    reversed_output = _event_output(model, reversed_frames[:, 0], device)
    query_output = _event_output(model, normal_frames[:, 2], device)
    remap = context ^ remap_flip

    normal_attempt = normal_output.logits.argmax(-1)
    reversed_attempt = reversed_output.logits.argmax(-1)
    normal_target = normal_batch.correct_actions[:, 0] ^ remap
    reversed_target = reversed_batch.correct_actions[:, 0] ^ remap
    attempts = torch.cat((
        normal_attempt, reversed_attempt,
        1 - normal_attempt, 1 - reversed_attempt))
    targets = torch.cat((
        normal_target, reversed_target,
        normal_target, reversed_target))
    strengths = (attempts == targets).to(torch.float32)
    keys = torch.cat((
        normal_output.memory_key, reversed_output.memory_key,
        normal_output.memory_key, reversed_output.memory_key))

    # The frozen output adapter defines the protocol intention for an opaque
    # action. The memory stores that intention; no new decoder is trained.
    prototypes = model.actuator.weight.detach()[attempts]
    values = torch.cat((
        prototypes,
        torch.zeros(
            prototypes.shape[0], model.width - model.intention_width,
            device=device, dtype=prototypes.dtype)), dim=-1)
    memory = DiskLatentMemory(
        width=model.width, capacity=keys.shape[0], device=device)
    memory.commit(keys, values, strengths, threshold=0.0)
    return memory, query_output.memory_key, remap


@torch.no_grad()
def _disk_actions(
        memory: DiskLatentMemory,
        query_keys: torch.Tensor,
        model: UnifiedCognitiveController,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return disk actions, values, and confidence after real save/reload."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "relation_memory.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        values, confidence = restored.retrieve(query_keys, top_k=1)
    actions = model.actuator(values[:, :model.intention_width]).argmax(-1)
    return actions, values, confidence


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = _digest(model)
    (
        normal_frames, reversed_frames, normal_batch, reversed_batch, context
    ) = _frames_and_batches(args.seed, args.contexts, device)

    memory, query_keys, remap = _memory_for_mapping(
        model, normal_frames, reversed_frames, normal_batch, reversed_batch,
        context, remap_flip=0, device=device)
    normal_actions, values, confidence = _disk_actions(
        memory, query_keys, model, device)
    normal_target = normal_batch.correct_actions[:, 2] ^ remap

    # Same frozen key/value path under a pixel-identical reversal of the
    # private context->action orientation.
    reversed_memory, reversed_keys, reversed_remap = _memory_for_mapping(
        model, normal_frames, reversed_frames, normal_batch, reversed_batch,
        context, remap_flip=1, device=device)
    reversed_actions, _, _ = _disk_actions(
        reversed_memory, reversed_keys, model, device)
    reversed_target = normal_batch.correct_actions[:, 2] ^ reversed_remap

    shuffled_actions = model.actuator(
        values.roll(1, dims=0)[:, :model.intention_width]).argmax(-1)
    corrupted_actions = model.actuator(
        values.flip(dims=(-1,))[:, :model.intention_width]).argmax(-1)
    null_action = torch.full(
        (args.contexts,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(args.contexts, device=device)
    query, _ = model.step(
        normal_frames[:, 2], model.initial_state(args.contexts, device=device),
        null_action, zeros, zeros)
    no_memory_actions = query.logits.argmax(-1)
    after = _digest(model)

    result: dict[str, object] = {
        "schema": "frozen-relation-memory-audit-v1",
        "checkpoint": str(args.checkpoint),
        "contexts": args.contexts,
        "device": str(device),
        "controller_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()),
        "weights_unchanged": before == after,
        "state_digest_before": before,
        "state_digest_after": after,
        "disk_accuracy": float(
            (normal_actions == normal_target).float().mean()),
        "reversed_accuracy": float(
            (reversed_actions == reversed_target).float().mean()),
        "prediction_flip_rate": float(
            (normal_actions != reversed_actions).float().mean()),
        "no_memory_accuracy": float(
            (no_memory_actions == normal_target).float().mean()),
        "shuffled_memory_accuracy": float(
            (shuffled_actions == normal_target).float().mean()),
        "corrupted_memory_accuracy": float(
            (corrupted_actions == normal_target).float().mean()),
        "confidence_mean": float(confidence.mean()),
        "stored_rows": memory.count,
        "claim_scope": (
            "frozen-controller visual relation plus generic external "
            "success-weighted episodic action memory; no trainable updates"),
    }
    gates = {
        "controller_frozen": bool(result["controller_frozen"]),
        "weights_unchanged": bool(result["weights_unchanged"]),
        "disk_memory_at_least_85": result["disk_accuracy"] >= 0.85,
        "reversed_at_least_85": result["reversed_accuracy"] >= 0.85,
        "prediction_flips_at_least_80": result["prediction_flip_rate"] >= 0.80,
        "no_memory_hurts": result["no_memory_accuracy"]
        <= result["disk_accuracy"] - 0.15,
        "shuffled_memory_hurts": result["shuffled_memory_accuracy"]
        <= result["disk_accuracy"] - 0.15,
        "corrupted_memory_hurts": result["corrupted_memory_accuracy"]
        <= result["disk_accuracy"] - 0.15,
    }
    result["gates"] = gates
    result["accepted"] = all(gates.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--device", default=_device_name())
    args = parser.parse_args()
    if args.contexts < 2 or args.contexts % 2:
        raise ValueError("contexts must be positive and even")
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
