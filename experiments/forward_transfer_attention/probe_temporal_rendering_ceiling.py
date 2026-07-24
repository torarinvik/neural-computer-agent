"""Supervised upper-bound probe for information visible in temporal task pixels.

This model is disposable: its weights are never loaded by the agent. It receives
the same mapping card, ordered object frames, and visual answer feedback available
to the agent, and tests whether those pixels determine the first/last rule.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import (COLORS, IMAGE_HEIGHT, IMAGE_WIDTH,
                          generate_temporal_attention_lifetime)
from .train import seed_everything


def _render_example(spec: tuple[int, bool, int, str]) -> tuple[np.ndarray, np.ndarray]:
    seed, heldout, mapping_line_width, feedback_mode = spec
    item = generate_temporal_attention_lifetime(
        seed, heldout=heldout, mapping_line_width=mapping_line_width,
        feedback_mode=feedback_mode)
    frames = np.concatenate((item.studies[0].frames, item.supports[0].frames), axis=0)
    if frames.shape[0] != 4:
        raise ValueError(f"expected four visible frames, received {frames.shape[0]}")
    first_identity = item.support_features[0][0]
    rewarded_identity = item.support_features[0][item.rule]
    return frames, np.asarray(
        (first_identity, rewarded_identity, item.rule), dtype=np.int64)


def _examples(start: int, count: int, *, heldout: bool,
              workers: int = 0, mapping_line_width: int = 4,
              feedback_mode: str = "white-button") -> tuple[torch.Tensor, torch.Tensor]:
    specs = ((start + index, heldout, mapping_line_width, feedback_mode)
             for index in range(count))
    images = np.empty((count, 4, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    labels = np.empty((count, 3), dtype=np.int64)
    if workers:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rendered = executor.map(_render_example, specs, chunksize=32)
            for index, (frames, label) in enumerate(rendered):
                images[index] = frames
                labels[index] = label
    else:
        for index, spec in enumerate(specs):
            frames, label = _render_example(spec)
            images[index] = frames
            labels[index] = label
    return (torch.from_numpy(images).permute(0, 1, 4, 2, 3),
            torch.from_numpy(labels))


def _pixel_oracle(item) -> int:
    """Recover the rule from rendered pixels alone; diagnostic, never agent input."""
    mapping = item.studies[0].frames[0]
    support = item.supports[0].frames
    button_x = np.asarray([
        int(round(18 + index * (IMAGE_WIDTH - 36) / 7)) for index in range(8)])
    mapped_actions = []
    observed_colors = []
    for frame in support[:2]:
        center_color = tuple(int(value) for value in frame[43, IMAGE_WIDTH // 2])
        color_id = min(range(2), key=lambda index: sum(
            abs(center_color[channel] - COLORS[index][channel]) for channel in range(3)))
        observed_colors.append(color_id)
    for color_id in range(2):
        color = np.asarray(COLORS[color_id], dtype=np.uint8)
        ys, xs = np.where(np.all(mapping == color, axis=-1) &
                          (np.indices(mapping.shape[:2])[0] >= 50))
        if not len(xs):
            raise ValueError("mapping line is not visible in rendered pixels")
        endpoint_x = float(xs[ys == ys.max()].mean())
        mapped_actions.append(int(np.abs(button_x - endpoint_x).argmin()))
    feedback = support[2]
    brightness = [float(feedback[79:86, x - 3:x + 4].mean()) for x in button_x]
    rewarded_action = int(np.argmax(brightness))
    first_action = mapped_actions[observed_colors[0]]
    second_action = mapped_actions[observed_colors[1]]
    if rewarded_action == first_action:
        return 0
    if rewarded_action == second_action:
        return 1
    raise ValueError("feedback does not match either rendered object")


class RenderingCeilingProbe(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4,
                 output_heads: int = 1) -> None:
        super().__init__()
        self.output_heads = output_heads
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=1, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 24 * 40, width), nn.LayerNorm(width),
        )
        self.frame_positions = nn.Parameter(torch.randn(1, 4, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 3, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.relation = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_heads * 2))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        encoded = self.frame_encoder(frames.reshape(batch * steps, *frames.shape[2:]))
        encoded = encoded.reshape(batch, steps, -1) + self.frame_positions[:, :steps]
        cls = self.cls.expand(batch, -1, -1)
        logits = self.head(self.relation(torch.cat((cls, encoded), dim=1))[:, 0])
        return logits.reshape(batch, self.output_heads, 2) if self.output_heads > 1 else logits


class PatchRelationProbe(nn.Module):
    """Preserve spatial tokens until after cross-frame relational processing."""
    def __init__(self, width: int = 96, heads: int = 4, patch: int = 8,
                 output_heads: int = 1) -> None:
        super().__init__()
        self.output_heads = output_heads
        self.patch = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        spatial_tokens = (96 // patch) * (160 // patch)
        self.spatial_positions = nn.Parameter(
            torch.randn(1, 1, spatial_tokens, width) * 0.02)
        self.frame_positions = nn.Parameter(torch.randn(1, 4, 1, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 3, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.relation = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(width))
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_heads * 2))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        patches = self.patch(frames.reshape(batch * steps, *frames.shape[2:]))
        patches = patches.flatten(2).transpose(1, 2).reshape(batch, steps, -1, patches.shape[1])
        patches = patches + self.spatial_positions + self.frame_positions[:, :steps]
        sequence = patches.flatten(1, 2)
        cls = self.cls.expand(batch, -1, -1)
        logits = self.head(self.relation(torch.cat((cls, sequence), dim=1))[:, 0])
        return logits.reshape(batch, self.output_heads, 2) if self.output_heads > 1 else logits


@torch.no_grad()
def _accuracies(model, frames, labels, batch_size, device) -> list[float]:
    model.eval()
    heads = labels.shape[1] if labels.ndim == 2 else 1
    correct = torch.zeros(heads, dtype=torch.long)
    for offset in range(0, labels.shape[0], batch_size):
        x = frames[offset:offset + batch_size].to(device, dtype=torch.float32).div_(255)
        y = labels[offset:offset + batch_size].to(device)
        predictions = model(x).argmax(-1)
        if predictions.ndim == 1:
            predictions, y = predictions[:, None], y[:, None]
        correct += (predictions.cpu() == y.cpu()).sum(0)
    return [int(value) / labels.shape[0] for value in correct]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--oracle-examples", type=int, default=4096)
    parser.add_argument("--generation-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--architecture", choices=("frame-vector", "patch-relation"),
                        default="frame-vector")
    parser.add_argument("--decomposed-heads", action="store_true")
    parser.add_argument("--target",
                        choices=("rule", "first-identity", "rewarded-identity"),
                        default="rule")
    parser.add_argument("--input-view", choices=("all", "mapping-feedback", "first-frame"),
                        default="all")
    parser.add_argument("--mapping-line-width", type=int, default=4)
    parser.add_argument(
        "--feedback-mode",
        choices=("white-button", "color-button"),
        default="white-button")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_x, train_y = _examples(
        17_000_000, args.train_examples, heldout=False, workers=args.generation_workers,
        mapping_line_width=args.mapping_line_width, feedback_mode=args.feedback_mode)
    test_x, test_y = _examples(
        19_000_000, args.test_examples, heldout=True, workers=args.generation_workers,
        mapping_line_width=args.mapping_line_width, feedback_mode=args.feedback_mode)
    if args.input_view == "mapping-feedback":
        train_x, test_x = train_x[:, (0, 3)], test_x[:, (0, 3)]
    elif args.input_view == "first-frame":
        train_x, test_x = train_x[:, 1:2], test_x[:, 1:2]
    output_heads = 3 if args.decomposed_heads else 1
    model = (PatchRelationProbe(output_heads=output_heads)
             if args.architecture == "patch-relation"
             else RenderingCeilingProbe(output_heads=output_heads)).to(device)
    if not args.decomposed_heads:
        target_index = {"first-identity": 0, "rewarded-identity": 1, "rule": 2}[args.target]
        train_y, test_y = train_y[:, target_index], test_y[:, target_index]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    total_steps = args.epochs * math.ceil(args.train_examples / args.batch_size)
    def learning_rate_factor(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return max(1e-3, (step + 1) / args.warmup_steps)
        if args.schedule == "constant":
            return 1.0
        progress = ((step - args.warmup_steps) /
                    max(1, total_steps - args.warmup_steps))
        return 0.05 + 0.95 * (1.0 + math.cos(math.pi * min(1.0, progress))) / 2.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    generator = torch.Generator().manual_seed(args.seed)
    history, best_train, best_test, optimizer_steps = [], 0.0, 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(args.train_examples, generator=generator)
        total_loss = 0.0
        for offset in range(0, args.train_examples, args.batch_size):
            indices = order[offset:offset + args.batch_size]
            x = train_x[indices].to(device, dtype=torch.float32).div_(255)
            y = train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = (nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
                    if logits.ndim == 3 else nn.functional.cross_entropy(logits, y))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * y.numel()
        if (epoch + 1) % args.eval_every_epochs and epoch + 1 != args.epochs:
            continue
        train_accuracies = _accuracies(
            model, train_x, train_y, args.batch_size, device)
        test_accuracies = _accuracies(model, test_x, test_y, args.batch_size, device)
        train_accuracy, test_accuracy = train_accuracies[-1], test_accuracies[-1]
        best_train = max(best_train, train_accuracy)
        best_test = max(best_test, test_accuracy)
        row = {"epoch": epoch + 1, "optimizer_steps": optimizer_steps,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "loss": total_loss / args.train_examples,
               "train_accuracy": train_accuracy, "test_accuracy": test_accuracy}
        if args.decomposed_heads:
            for index, name in enumerate(("first_identity", "rewarded_identity", "rule")):
                row[f"train_{name}_accuracy"] = train_accuracies[index]
                row[f"test_{name}_accuracy"] = test_accuracies[index]
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    oracle_accuracy = None
    if args.oracle_examples:
        oracle_accuracy = sum(
            _pixel_oracle(item) == item.rule
            for item in (generate_temporal_attention_lifetime(
                21_000_000 + index, heldout=True)
                         for index in range(args.oracle_examples))) / args.oracle_examples
    curve = [(0, 0.0)] + [(row["optimizer_steps"], max(
        0.0, min(1.0, (row["test_accuracy"] - 0.5) / 0.5))) for row in history]
    normalized_area = sum(
        (right_step - left_step) * (left_score + right_score) / 2
        for (left_step, left_score), (right_step, right_score)
        in zip(curve, curve[1:])) / max(1, optimizer_steps)
    report = {
        "schema": "temporal-rendering-ceiling-v1",
        "disposable_supervised_probe": True,
        "inputs": "mapping-card-plus-one-visual-support",
        "game_state_inputs": False,
        "architecture": args.architecture,
        "decomposed_heads": args.decomposed_heads,
        "target": args.target,
        "input_view": args.input_view,
        "mapping_line_width": args.mapping_line_width,
        "feedback_mode": args.feedback_mode,
        "schedule": args.schedule,
        "warmup_steps": args.warmup_steps,
        "train_examples": args.train_examples,
        "test_examples": args.test_examples,
        "train_rule_rate": float(
            (train_y[:, -1] if train_y.ndim == 2 else train_y).float().mean()),
        "test_rule_rate": float(
            (test_y[:, -1] if test_y.ndim == 2 else test_y).float().mean()),
        "best_train_accuracy": best_train,
        "best_test_accuracy": best_test,
        "heldout_early_learning_auc_above_chance": normalized_area,
        "pixel_oracle_examples": args.oracle_examples,
        "pixel_oracle_accuracy": oracle_accuracy,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
