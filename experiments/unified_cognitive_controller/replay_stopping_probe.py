"""Probe whether marginal replay value transfers across task generations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything


FEATURES = (
    "loss_before",
    "previous_loss_reduction",
    "log_observed_examples",
    "previous_gradient_norm",
    "replay_fraction",
)


def trace_features(
        rows: list[dict[str, object]], replay_updates: int,
) -> torch.Tensor:
    """Return decision-time features only; no future outcome is included."""
    return torch.tensor([
        [
            float(row["loss_before"]),
            float(row["previous_loss_reduction"]),
            math.log1p(int(row["observed_examples"])),
            float(row["previous_gradient_norm"]),
            int(row["replay_index"]) / max(1, replay_updates - 1),
        ]
        for row in rows
    ], dtype=torch.float32)


def trace_targets(rows: list[dict[str, object]]) -> torch.Tensor:
    return torch.tensor(
        [float(row["loss_reduction"]) for row in rows],
        dtype=torch.float32,
    )[:, None]


class ReplayBenefitProbe(nn.Module):
    def __init__(self, width: int = len(FEATURES), hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def save_probe(
        path: Path, model: ReplayBenefitProbe, *,
        feature_mean: torch.Tensor, feature_scale: torch.Tensor,
        target_mean: torch.Tensor, target_scale: torch.Tensor,
        hidden: int, target_horizon: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "replay-benefit-probe-v1",
        "features": FEATURES,
        "hidden": hidden,
        "target_horizon": target_horizon,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "target_mean": target_mean.detach().cpu(),
        "target_scale": target_scale.detach().cpu(),
    }, path)


def load_probe(
        path: Path, device: torch.device,
) -> tuple[ReplayBenefitProbe, dict[str, torch.Tensor | int]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "replay-benefit-probe-v1":
        raise ValueError("unsupported replay-benefit checkpoint")
    if tuple(payload["features"]) != FEATURES:
        raise ValueError("replay-benefit feature schema mismatch")
    model = ReplayBenefitProbe(hidden=int(payload["hidden"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalization = {
        key: payload[key].to(device)
        for key in (
            "feature_mean", "feature_scale",
            "target_mean", "target_scale")
    }
    normalization["target_horizon"] = int(
        payload.get("target_horizon", 1))
    return model, normalization


@torch.no_grad()
def predict_replay_benefit(
        model: ReplayBenefitProbe,
        normalization: dict[str, torch.Tensor | int], *,
        loss_before: float,
        previous_loss_reduction: float,
        previous_gradient_norm: float,
        observed_examples: int,
        replay_index: int,
        replay_updates: int,
        device: torch.device,
) -> float:
    features = trace_features([{
        "loss_before": loss_before,
        "previous_loss_reduction": previous_loss_reduction,
        "previous_gradient_norm": previous_gradient_norm,
        "observed_examples": observed_examples,
        "replay_index": replay_index,
    }], replay_updates).to(device)
    feature_mean = normalization["feature_mean"]
    feature_scale = normalization["feature_scale"]
    target_mean = normalization["target_mean"]
    target_scale = normalization["target_scale"]
    assert isinstance(feature_mean, torch.Tensor)
    assert isinstance(feature_scale, torch.Tensor)
    assert isinstance(target_mean, torch.Tensor)
    assert isinstance(target_scale, torch.Tensor)
    normalized = (features - feature_mean) / feature_scale
    prediction = (
        model(normalized) * target_scale + target_mean)
    return float(prediction.squeeze())


def load_trace(
        path: Path, arm: str, target_horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = json.loads(path.read_text())
    replay_updates = int(payload["configuration"]["replay_updates"])
    all_rows = [
        row for row in payload["replay_trace"]
        if row["arm"] == arm
    ]
    if not all_rows:
        raise ValueError(f"{path} has no replay trace rows for {arm}")
    rows = []
    targets = []
    for experience_step in sorted({
            int(row["experience_step"]) for row in all_rows}):
        group = sorted(
            (row for row in all_rows
             if int(row["experience_step"]) == experience_step),
            key=lambda row: int(row["replay_index"]))
        for start in range(0, len(group), target_horizon):
            end = start + target_horizon
            if end > len(group):
                continue
            rows.append(group[start])
            targets.append(
                float(group[start]["loss_before"])
                - float(group[end - 1]["loss_after"]))
    return (
        trace_features(rows, replay_updates),
        torch.tensor(targets, dtype=torch.float32)[:, None],
        replay_updates,
    )


def regression_metrics(
        prediction: torch.Tensor, target: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.flatten()
    target = target.flatten()
    error = prediction - target
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    )
    correlation = (
        float((centered_prediction * centered_target).sum() / denominator)
        if float(denominator) > 0 else 0.0
    )
    baseline_mae = float((target - target.mean()).abs().mean())
    mae = float(error.abs().mean())
    target_positive = target > 0
    predicted_positive = prediction > 0
    return {
        "mae": mae,
        "mean_baseline_mae": baseline_mae,
        "mae_improvement_fraction": (
            1.0 - mae / baseline_mae if baseline_mae > 0 else 0.0),
        "pearson_correlation": correlation,
        "positive_rate": float(target_positive.float().mean()),
        "sign_accuracy": float(
            (target_positive == predicted_positive).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--test-report", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=("composition", "flat"),
                        default="composition")
    parser.add_argument("--seed", type=int, default=8110)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    if args.target_horizon < 1:
        raise ValueError("target horizon must be positive")
    train_parts = [load_trace(path, args.arm, args.target_horizon)[:2]
                   for path in args.train_report]
    test_parts = [load_trace(path, args.arm, args.target_horizon)[:2]
                  for path in args.test_report]
    train_x = torch.cat([part[0] for part in train_parts]).to(device)
    train_y = torch.cat([part[1] for part in train_parts]).to(device)
    test_x = torch.cat([part[0] for part in test_parts]).to(device)
    test_y = torch.cat([part[1] for part in test_parts]).to(device)

    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    target_mean = train_y.mean(0, keepdim=True)
    target_scale = train_y.std(0, keepdim=True).clamp_min(1e-8)
    model = ReplayBenefitProbe(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    batch_size = min(256, train_x.shape[0])
    for _ in range(args.steps):
        indices = torch.randint(
            0, train_x.shape[0], (batch_size,),
            generator=generator, device=device)
        prediction = model((train_x[indices] - mean) / scale)
        normalized_target = (
            train_y[indices] - target_mean) / target_scale
        loss = nn.functional.mse_loss(prediction, normalized_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = (
            model((train_x - mean) / scale) * target_scale + target_mean)
        test_prediction = (
            model((test_x - mean) / scale) * target_scale + target_mean)
        shuffled = test_y[torch.randperm(
            test_y.shape[0], generator=generator, device=device)]
    report = {
        "schema": "cross-generation-replay-benefit-probe-v1",
        "configuration": {
            "train_reports": [str(path) for path in args.train_report],
            "test_reports": [str(path) for path in args.test_report],
            "arm": args.arm,
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "hidden": args.hidden,
            "target_horizon": args.target_horizon,
            "features": FEATURES,
        },
        "train_rows": int(train_x.shape[0]),
        "test_rows": int(test_x.shape[0]),
        "train": regression_metrics(train_prediction, train_y),
        "held_out_generation": regression_metrics(test_prediction, test_y),
        "shuffled_target_control": regression_metrics(test_prediction, shuffled),
        "gate": {
            "held_out_beats_mean_mae_by_10_percent":
                regression_metrics(test_prediction, test_y)[
                    "mae_improvement_fraction"] >= 0.10,
            "held_out_correlation_at_least_0_25":
                regression_metrics(test_prediction, test_y)[
                    "pearson_correlation"] >= 0.25,
            "shuffled_correlation_below_0_10":
                abs(regression_metrics(test_prediction, shuffled)[
                    "pearson_correlation"]) < 0.10,
        },
    }
    report["gate"]["accepted_for_policy_integration"] = all(
        report["gate"].values())
    if args.checkpoint is not None:
        save_probe(
            args.checkpoint, model,
            feature_mean=mean, feature_scale=scale,
            target_mean=target_mean, target_scale=target_scale,
            hidden=args.hidden, target_horizon=args.target_horizon)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
