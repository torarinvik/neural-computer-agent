from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import EpisodeDataset, collate_episodes
from .model import LatentAgent, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def supervised_loss(output, batch: dict[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
    """Keep final-answer learning dominant over the repeated NEXT targets."""
    logits = output.logits
    actions = batch["actions"]
    mask = batch["mask"]
    all_actions = criterion(logits.reshape(-1, logits.shape[-1]), actions.reshape(-1))
    final_indices = mask.sum(1) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    final_answers = criterion(logits[rows, final_indices], actions[rows, final_indices])
    entity_count = output.subject_logits.shape[-1]
    subject_loss = criterion(output.subject_logits.reshape(-1, entity_count), batch["subjects"].reshape(-1))
    relation_loss = criterion(output.relation_logits.reshape(-1, 8), batch["relations"].reshape(-1))
    object_loss = criterion(output.object_logits.reshape(-1, entity_count), batch["objects"].reshape(-1))
    final_targets = torch.zeros_like(output.final_logits)
    final_targets[torch.arange(mask.shape[0], device=mask.device), mask.sum(1) - 1] = 1.0
    final_loss = nn.functional.binary_cross_entropy_with_logits(output.final_logits[mask],
                                                                 final_targets[mask])
    return (0.25 * all_actions + final_answers
            + 0.5 * (subject_loss + relation_loss + object_loss) + 0.25 * final_loss)


@torch.no_grad()
def evaluate(model: LatentAgent, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    total = final_right = action_right = action_count = 0
    subject_right = relation_right = object_right = final_detection_right = 0
    model_seconds = 0.0
    by_cards: dict[int, list[int]] = {}
    confusion_by_cards: dict[int, dict[str, int]] = {}
    started = time.perf_counter()
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        pcm = batch["pcm"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        subjects = batch["subjects"].to(device, non_blocking=True)
        relations = batch["relations"].to(device, non_blocking=True)
        objects = batch["objects"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(frames, pcm, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - model_started
        predicted = output.logits.argmax(-1)
        action_right += int(((predicted == actions) & mask).sum())
        action_count += int(mask.sum())
        subject_right += int(((output.subject_logits.argmax(-1) == subjects) & mask).sum())
        relation_right += int(((output.relation_logits.argmax(-1) == relations) & mask).sum())
        object_right += int(((output.object_logits.argmax(-1) == objects) & mask).sum())
        rows = torch.arange(frames.shape[0], device=device)
        final_targets = torch.zeros_like(mask)
        final_targets[rows, mask.sum(1) - 1] = True
        final_detection_right += int((((output.final_logits > 0) == final_targets) & mask).sum())
        lengths = mask.sum(1) - 1
        final_matches = predicted[rows, lengths] == actions[rows, lengths]
        final_right += int(final_matches.sum())
        for cards, matched in zip((lengths + 1).tolist(), final_matches.tolist()):
            bucket = by_cards.setdefault(int(cards), [0, 0])
            bucket[0] += int(matched)
            bucket[1] += 1
        final_targets = actions[rows, lengths]
        final_predictions = predicted[rows, lengths]
        for cards, target, prediction in zip((lengths + 1).tolist(),
                                              final_targets.tolist(),
                                              final_predictions.tolist()):
            bucket = confusion_by_cards.setdefault(int(cards), {
                "episodes": 0, "target_true": 0, "predicted_true": 0,
                "correct": 0,
            })
            bucket["episodes"] += 1
            bucket["target_true"] += int(target == 3)
            bucket["predicted_true"] += int(prediction == 3)
            bucket["correct"] += int(target == prediction)
        total += frames.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"episodes": total, "final_accuracy": final_right / max(1, total),
            "action_accuracy": action_right / max(1, action_count),
            "subject_accuracy": subject_right / max(1, action_count),
            "relation_accuracy": relation_right / max(1, action_count),
            "object_accuracy": object_right / max(1, action_count),
            "final_detection_accuracy": final_detection_right / max(1, action_count),
            "end_to_end_milliseconds_per_episode": elapsed * 1000 / max(1, total),
            "model_milliseconds_per_episode": model_seconds * 1000 / max(1, total),
            "model_milliseconds_per_event": model_seconds * 1000 / max(1, action_count),
            "final_accuracy_by_cards": {
                str(cards): right / count for cards, (right, count) in sorted(by_cards.items())
            },
            "final_confusion_by_cards": {
                str(cards): values for cards, values in sorted(confusion_by_cards.items())
            }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", choices=("gru", "graph", "graph_cached", "closure", "recursive"), default="gru")
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--recursive-steps", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--eval-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-premises", default="2,3,4,5,6")
    parser.add_argument("--eval-premises", default="2,4,6,8,16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--entity-count", type=int, default=64)
    parser.add_argument("--length-curriculum", action="store_true")
    parser.add_argument("--no-positions", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_premises = tuple(map(int, args.train_premises.split(",")))
    eval_premises = tuple(map(int, args.eval_premises.split(",")))
    validation_data = EpisodeDataset(args.eval_samples, start_seed=80_000,
                                     premise_choices=train_premises, heldout=True,
                                     entity_count=args.entity_count)
    generalization_data = EpisodeDataset(args.eval_samples, start_seed=100_000,
                                         premise_choices=eval_premises,
                                         heldout=True, final=True,
                                         entity_count=args.entity_count)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "collate_fn": collate_episodes, "pin_memory": device.type == "cuda"}
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    generalization_loader = DataLoader(generalization_data, shuffle=False, **loader_options)
    model = LatentAgent(args.core, args.hidden, args.recursive_steps,
                        entity_count=args.entity_count,
                        use_positions=not args.no_positions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        if args.length_curriculum:
            stage = min(len(train_premises), max(1, (epoch + 1) * len(train_premises) // args.epochs))
            epoch_premises = train_premises[:stage]
        else:
            epoch_premises = train_premises
        train_data = EpisodeDataset(args.train_samples,
                                    start_seed=0,
                                    premise_choices=epoch_premises,
                                    entity_count=args.entity_count)
        train_loader = DataLoader(train_data, shuffle=True, **loader_options)
        model.train()
        loss_total = 0.0
        batches = 0
        for batch in train_loader:
            frames = batch["frames"].to(device, non_blocking=True)
            pcm = batch["pcm"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            training_batch = {key: value.to(device, non_blocking=True)
                              for key, value in batch.items()
                              if key in {"actions", "subjects", "relations", "objects", "mask"}}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(frames, pcm, mask)
                loss = supervised_loss(output, training_batch, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.detach())
            batches += 1
        validation = evaluate(model, validation_loader, device)
        history.append({"epoch": epoch + 1, "premise_choices": epoch_premises,
                        "loss": loss_total / max(1, batches),
                        "validation": validation})
        print(json.dumps(history[-1]), flush=True)
    training_seconds = time.perf_counter() - training_started
    generalization = evaluate(model, generalization_loader, device)
    metadata = {"schema": "syllogimous-latent-agent-checkpoint-v1", "core": args.core,
                "hidden": args.hidden, "recursive_steps": args.recursive_steps,
                "entity_count": args.entity_count,
                "length_curriculum": args.length_curriculum,
                "use_positions": not args.no_positions,
                "parameters": parameter_count(model), "seed": args.seed,
                "train_premises": train_premises, "eval_premises": eval_premises,
                "history": history, "generalization": generalization,
                "training_seconds": training_seconds}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, args.checkpoint)
    args.report.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "report": str(args.report),
                      "parameters": metadata["parameters"], "generalization": generalization}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
