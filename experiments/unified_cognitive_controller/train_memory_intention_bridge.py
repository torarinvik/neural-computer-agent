"""Train and audit a label-free memory-code/intention bridge.

The controller core never updates.  Bridge training uses only a support
attempt, its scalar verifier outcome, and the frozen controller's own query
action.  The true query answer is retained exclusively for held-out auditing.
After bridge training, only disk memory rows change in the adaptation arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .memory_intention_bridge import MemoryActionComposer, MemoryCodeBridge
from .model import UnifiedCognitiveController


def _digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _collect(
        model: UnifiedCognitiveController, *, seed: int, remap_seed: int,
        count: int, remap_flip: int, device: torch.device,
        ) -> dict[str, torch.Tensor]:
    """Collect learner-visible support/query tensors and private audit facts."""
    batch = generate_lifetimes(
        count, 3, seed=seed, heldout=True, task="pair_relation",
        appearance="bars", support_trials=1, device=device)
    # Each context gets a unique visual code.  It is independent of the
    # private mapping bit and becomes an external, amodal memory key.
    code_generator = torch.Generator().manual_seed(seed + 800_000)
    codes = (
        0.10 + 0.80 * torch.rand(
            count, 3, 3, 8, generator=code_generator)
    ).to(device)
    frames = batch.frames.clone()
    frames[:, :, :, :3, :8] = codes.unsqueeze(1)
    remap_generator = torch.Generator().manual_seed(remap_seed)
    remap = (
        (torch.arange(count) % 2)[
            torch.randperm(count, generator=remap_generator)]
        .to(device)
        ^ remap_flip
    )
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    state = model.initial_state(count, device=device)
    support0, state = model.step(
        frames[:, 0], state, null_action, zeros, zeros)
    attempted = support0.logits.argmax(-1)
    support_reward = (
        attempted == (batch.correct_actions[:, 0] ^ remap)
    ).to(torch.float32)
    support1, _ = model.step(
        frames[:, 1], state, attempted, support_reward,
        torch.ones_like(support_reward))
    query, _ = model.step(
        frames[:, 2], model.initial_state(count, device=device),
        null_action, zeros, zeros)
    # This target is learner-visible: it is the frozen query action corrected
    # by the success/failure bit from the support.  It is not the verifier's
    # query label and is never used for the final capability score.
    pseudo_action = (
        query.logits.argmax(-1) ^ (1 - support_reward).long())
    key_patch = frames[:, 0, :, :3, :8].flatten(1)
    query_patch = frames[:, 2, :, :3, :8].flatten(1)
    padding = model.width - key_patch.shape[-1]
    keys = torch.cat((
        key_patch, torch.zeros(count, padding, device=device)), dim=-1)
    query_keys = torch.cat((
        query_patch, torch.zeros(count, padding, device=device)), dim=-1)
    return {
        "memory_values": support1.memory_value,
        "query_intentions": query.intention,
        "keys": keys,
        "query_keys": query_keys,
        "support_reward": support_reward,
        "pseudo_actions": pseudo_action,
        # Private scoring data is kept separate and never enters training.
        "query_targets": batch.correct_actions[:, 2] ^ remap,
        "remap": remap,
        "frames": frames,
    }


@torch.no_grad()
def _disk_read(
        model: UnifiedCognitiveController, bridge: MemoryCodeBridge,
        composer: MemoryActionComposer, data: dict[str, torch.Tensor],
        device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = bridge(data["memory_values"])
    padded_values = torch.cat((
        values,
        torch.zeros(
            values.shape[0], model.width - values.shape[-1], device=device)),
        dim=-1)
    memory = DiskLatentMemory(
        width=model.width, capacity=values.shape[0], device=device)
    memory.commit(
        data["keys"], padded_values,
        torch.ones(values.shape[0], device=device), threshold=0.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        retrieved, confidence = restored.retrieve(
            data["query_keys"], top_k=1)
    actions = composer(
        data["query_intentions"], retrieved[:, :2]).argmax(-1)
    return actions, retrieved, confidence


def _accuracy(actions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((actions == targets).float().mean())


@torch.no_grad()
def _audit(
        model: UnifiedCognitiveController, bridge: MemoryCodeBridge,
        composer: MemoryActionComposer, normal: dict[str, torch.Tensor],
        reversed_data: dict[str, torch.Tensor], device: torch.device,
        ) -> dict[str, object]:
    normal_actions, retrieved, confidence = _disk_read(
        model, bridge, composer, normal, device)
    reversed_actions, _, _ = _disk_read(
        model, bridge, composer, reversed_data, device)
    shuffled_actions = composer(
        normal["query_intentions"], retrieved.roll(1, dims=0)[:, :2]
    ).argmax(-1)
    corrupted_actions = composer(
        normal["query_intentions"], retrieved.flip(dims=(-1,))[:, :2]
    ).argmax(-1)
    null_action = torch.full(
        (normal["frames"].shape[0],), NULL_ACTION,
        dtype=torch.long, device=device)
    zeros = torch.zeros(normal["frames"].shape[0], device=device)
    query, _ = model.step(
        normal["frames"][:, 2],
        model.initial_state(normal["frames"].shape[0], device=device),
        null_action, zeros, zeros)
    no_memory = query.logits.argmax(-1)
    return {
        "disk_accuracy": _accuracy(
            normal_actions, normal["query_targets"]),
        "reversed_accuracy": _accuracy(
            reversed_actions, reversed_data["query_targets"]),
        "prediction_flip_rate": float(
            (normal_actions != reversed_actions).float().mean()),
        "no_memory_accuracy": _accuracy(
            no_memory, normal["query_targets"]),
        "shuffled_memory_accuracy": _accuracy(
            shuffled_actions, normal["query_targets"]),
        "corrupted_memory_accuracy": _accuracy(
            corrupted_actions, normal["query_targets"]),
        "retrieval_confidence_mean": float(confidence.mean()),
        "retrieval_exact_rate": float(
            (
                torch.nn.functional.normalize(normal["query_keys"], dim=-1)
                @ torch.nn.functional.normalize(normal["keys"], dim=-1).T
            ).argmax(-1).eq(
                torch.arange(normal["keys"].shape[0], device=device)
            ).float().mean()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    payload = torch.load(
        args.controller, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller_before = _digest(model)
    train = _collect(
        model, seed=args.seed, remap_seed=args.seed + 100_000,
        count=args.train_contexts, remap_flip=0, device=device)
    test = _collect(
        model, seed=args.seed + 1, remap_seed=args.seed + 200_000,
        count=args.test_contexts, remap_flip=0, device=device)
    reversed_test = _collect(
        model, seed=args.seed + 1, remap_seed=args.seed + 200_000,
        count=args.test_contexts, remap_flip=1, device=device)

    bridge = MemoryCodeBridge(model.width).to(device)
    composer = MemoryActionComposer(model.intention_width).to(device)
    optimizer = torch.optim.AdamW(
        list(bridge.parameters()) + list(composer.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 300_000)
    curve = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, args.train_contexts, (min(512, args.train_contexts),),
            generator=generator)
        indices = indices.to(device)
        code = bridge(train["memory_values"][indices])
        code_loss = torch.nn.functional.cross_entropy(
            code, train["support_reward"][indices].long())
        action_logits = composer(
            train["query_intentions"][indices], code)
        action_loss = torch.nn.functional.cross_entropy(
            action_logits, train["pseudo_actions"][indices])
        loss = code_loss + action_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(bridge.parameters()) + list(composer.parameters()), 1.0)
        optimizer.step()
        if step in {1, args.steps} or step % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                curve.append({
                    "step": step,
                    "loss": float(loss),
                    "code_loss": float(code_loss),
                    "pseudo_action_accuracy": float(
                        (action_logits.argmax(-1)
                         == train["pseudo_actions"][indices]).float().mean()),
                })
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    for parameter in composer.parameters():
        parameter.requires_grad_(False)
    report = _audit(
        model, bridge, composer, test, reversed_test, device)
    controller_after = _digest(model)
    report.update({
        "schema": "memory-intention-bridge-audit-v1",
        "controller": str(args.controller),
        "controller_digest_before": controller_before,
        "controller_digest_after": controller_after,
        "controller_weights_unchanged": controller_before == controller_after,
        "bridge_frozen_during_adaptation": all(
            not parameter.requires_grad for parameter in bridge.parameters()),
        "composer_frozen_during_adaptation": all(
            not parameter.requires_grad for parameter in composer.parameters()),
        "train_contexts": args.train_contexts,
        "test_contexts": args.test_contexts,
        "seed": args.seed,
        "device": str(device),
        "support_verifier_bits": args.train_contexts,
        "optimizer_steps": args.steps,
        "learning_rate": args.learning_rate,
        "curve": curve,
        "bridge_parameters": sum(
            parameter.numel() for parameter in bridge.parameters()),
        "composer_parameters": sum(
            parameter.numel() for parameter in composer.parameters()),
        "labels_used_for_training": [],
        "learner_visible_training_signal": [
            "attempted opaque support action",
            "scalar support success/failure",
            "frozen controller query action",
        ],
        "private_query_targets_used_only_for_audit": True,
        "wall_seconds": time.perf_counter() - started,
    })
    report["gates"] = {
        "controller_frozen": report["controller_weights_unchanged"],
        "bridge_frozen": report["bridge_frozen_during_adaptation"],
        "composer_frozen": report["composer_frozen_during_adaptation"],
        "disk_at_least_85": report["disk_accuracy"] >= 0.85,
        "reversal_at_least_85": report["reversed_accuracy"] >= 0.85,
        "prediction_flips_at_least_80":
            report["prediction_flip_rate"] >= 0.80,
        "no_memory_hurts": report["no_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
        "shuffled_memory_hurts": report["shuffled_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
        "corrupted_memory_hurts": report["corrupted_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
        "exact_disk_retrieval": report["retrieval_exact_rate"] >= 0.99,
    }
    report["accepted"] = all(report["gates"].values())
    args.bridge_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "memory-intention-bridge-v1",
        "controller": str(args.controller),
        "memory_width": model.width,
        "intention_width": model.intention_width,
        "bridge_state_dict": {
            name: value.detach().cpu()
            for name, value in bridge.state_dict().items()},
        "composer_state_dict": {
            name: value.detach().cpu()
            for name, value in composer.state_dict().items()},
        "training": {
            "support_verifier_bits": args.train_contexts,
            "optimizer_steps": args.steps,
            "labels_used": [],
        },
    }, args.bridge_out)
    report["bridge_checkpoint"] = str(args.bridge_out)
    report["bridge_checkpoint_sha256"] = _sha256(args.bridge_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--bridge-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-contexts", type=int, default=256)
    parser.add_argument("--test-contexts", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=29001)
    parser.add_argument("--device", default=_device_name())
    args = parser.parse_args()
    if args.train_contexts < 2 or args.train_contexts % 2:
        raise ValueError("train contexts must be positive and even")
    if args.test_contexts < 2 or args.test_contexts % 2:
        raise ValueError("test contexts must be positive and even")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
