"""Probe both ends of the controller's dormant persistent-memory interface.

The controller is frozen. Verifier-private labels are used only by disposable
diagnostic heads and never enter an agent checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController
from .train import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_context_signatures(
        batch: CognitiveLifetimeBatch, *, seed: int
        ) -> CognitiveLifetimeBatch:
    """Paint one stable, nonsemantic visual signature into every session."""
    generator = torch.Generator().manual_seed(seed)
    # A 3x3 RGB code gives each recurring world a high-entropy sensory key.
    codes = (
        0.25 + 0.70 * torch.rand(
            batch.batch_size, 3, 3, 3, generator=generator)
    ).to(batch.frames.device)
    frames = batch.frames.clone()
    frames[:, :, :, 2:5, 2:5] = codes.unsqueeze(1)
    return CognitiveLifetimeBatch(
        frames=frames,
        correct_actions=batch.correct_actions,
        stimulus_identities=batch.stimulus_identities,
        rule_bits=batch.rule_bits,
        seeds=batch.seeds)


@torch.no_grad()
def _extract(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        heldout: bool, device: torch.device) -> dict[str, torch.Tensor]:
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output0, state = model.step(
        batch.frames[:, 0], state, null_action, zeros, zeros)
    attempted = output0.logits.argmax(-1)
    outcome = (
        attempted == batch.correct_actions[:, 0]).to(torch.float32)
    output1, _ = model.step(
        batch.frames[:, 1], state, attempted, outcome,
        torch.ones_like(outcome))

    fresh = model.initial_state(count, device=device)
    query, _ = model.step(
        batch.frames[:, 2], fresh, null_action, zeros, zeros)
    oracle, _ = model.step(
        batch.frames[:, 2], fresh, null_action, zeros, zeros,
        retrieved_memory=output1.memory_value)
    shuffled, _ = model.step(
        batch.frames[:, 2], fresh, null_action, zeros, zeros,
        retrieved_memory=output1.memory_value.roll(1, dims=0))
    return {
        "pre_feedback_key": output0.memory_key,
        "post_feedback_key": output1.memory_key,
        "post_feedback_value": output1.memory_value,
        "query_key": query.memory_key,
        "rules": batch.rule_bits,
        "query_correct_actions": batch.correct_actions[:, 2],
        "no_memory_actions": query.logits.argmax(-1),
        "oracle_memory_actions": oracle.logits.argmax(-1),
        "shuffled_memory_actions": shuffled.logits.argmax(-1),
    }


def _retrieval(
        queries: torch.Tensor, keys: torch.Tensor) -> dict[str, float]:
    queries = nn.functional.normalize(queries, dim=-1)
    keys = nn.functional.normalize(keys, dim=-1)
    similarity = queries @ keys.T
    ranking = similarity.topk(k=min(4, keys.shape[0]), dim=-1).indices
    target = torch.arange(keys.shape[0], device=keys.device)
    return {
        "top1": float((ranking[:, 0] == target).float().mean()),
        "top4": float(
            (ranking == target.unsqueeze(1)).any(dim=1).float().mean()),
        "shuffled_pair_top1": float(
            (ranking[:, 0].roll(1) == target).float().mean()),
    }


def _retrieval_curve(
        queries: torch.Tensor, keys: torch.Tensor
        ) -> dict[str, dict[str, float]]:
    curve = {}
    for count in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        if count <= keys.shape[0]:
            curve[str(count)] = _retrieval(
                queries[:count], keys[:count])
    return curve


def _accuracy(
        actions: torch.Tensor, correct: torch.Tensor) -> float:
    return float((actions == correct).float().mean())


def _fit_rule_probe(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        seed: int, shuffled: bool) -> dict[str, float]:
    generator = torch.Generator(device=train_x.device).manual_seed(seed)
    labels = train_y
    if shuffled:
        labels = labels[torch.randperm(
            labels.numel(), generator=generator, device=labels.device)]
    probe = nn.Sequential(
        nn.LayerNorm(train_x.shape[-1]),
        nn.Linear(train_x.shape[-1], 64),
        nn.GELU(),
        nn.Linear(64, 2),
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        indices = torch.randint(
            0, train_x.shape[0], (512,), generator=generator,
            device=train_x.device)
        loss = nn.functional.cross_entropy(
            probe(train_x[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {
            "train_accuracy": float(
                (probe(train_x).argmax(-1) == labels).float().mean()),
            "heldout_accuracy": float(
                (probe(test_x).argmax(-1) == test_y).float().mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--contexts", type=int, default=1024)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.contexts % 2:
        raise ValueError("contexts must be even")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train = _extract(
        model, count=args.contexts, seed=args.seed,
        heldout=False, device=device)
    test = _extract(
        model, count=args.contexts, seed=args.seed + 1,
        heldout=True, device=device)
    report = {
        "schema": "unified-controller-persistent-interface-probe-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "contexts_per_split": args.contexts,
        "diagnostic_only": True,
        "agent_weights_changed": False,
        "probe_weights_discarded": True,
        "verifier_private_labels_used_by_probe": True,
        "query_to_pre_feedback_key": _retrieval(
            test["query_key"], test["pre_feedback_key"]),
        "pre_feedback_key_scaling": _retrieval_curve(
            test["query_key"], test["pre_feedback_key"]),
        "query_to_post_feedback_key": _retrieval(
            test["query_key"], test["post_feedback_key"]),
        "read_path_behavior": {
            "no_memory": _accuracy(
                test["no_memory_actions"],
                test["query_correct_actions"]),
            "oracle_same_context_value": _accuracy(
                test["oracle_memory_actions"],
                test["query_correct_actions"]),
            "shuffled_context_value": _accuracy(
                test["shuffled_memory_actions"],
                test["query_correct_actions"]),
        },
        "rule_in_post_feedback_value": _fit_rule_probe(
            train["post_feedback_value"], train["rules"],
            test["post_feedback_value"], test["rules"],
            seed=args.seed + 2, shuffled=False),
        "shuffled_rule_control": _fit_rule_probe(
            train["post_feedback_value"], train["rules"],
            test["post_feedback_value"], test["rules"],
            seed=args.seed + 3, shuffled=True),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
