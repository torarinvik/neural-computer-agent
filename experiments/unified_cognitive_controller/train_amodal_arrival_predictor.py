"""Train a generic arrival predictor from observed transport traces only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .amodal_wait_policy import AMODAL_WAIT_FEATURES, AmodalArrivalPredictor


def _trace_batch(
    batch_size: int,
    *,
    seed: int,
    history_size: int,
    deadline: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate self-supervised rows from event arrival traces.

    The two transport regimes are intentionally latent: a trace has either a
    sparse or a reliable arrival rate, and the predictor sees only recent
    arrivals, pending age, and generic cardinality. The target is the next
    observed arrival, not a task or modality label.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    reliable = torch.randint(0, 2, (batch_size,), generator=generator, device=device)
    rate = torch.where(
        reliable.bool(),
        torch.full((batch_size,), 0.82, device=device),
        torch.full((batch_size,), 0.18, device=device),
    )
    history = torch.rand(
        batch_size, history_size, generator=generator, device=device
    ) < rate[:, None]
    delay = torch.where(
        torch.rand(batch_size, generator=generator, device=device) < rate,
        torch.randint(
            1,
            int(deadline) + 1,
            (batch_size,),
            generator=generator,
            device=device,
        ),
        torch.full((batch_size,), int(deadline) + 1, device=device),
    )
    last_gap = torch.full(
        (batch_size,), history_size + 1, dtype=torch.float32, device=device
    )
    for index in range(history_size):
        observed = history[:, -(index + 1)]
        last_gap = torch.where(
            (last_gap == history_size + 1) & observed,
            torch.full_like(last_gap, index + 1),
            last_gap,
        )
    present_fraction = torch.randint(
        1, 4, (batch_size,), generator=generator, device=device
    ).float() / 4.0
    age = torch.randint(
        0,
        int(deadline),
        (batch_size,),
        generator=generator,
        device=device,
    ).float()
    features = torch.stack(
        [
            present_fraction,
            (age / deadline).clamp(0.0, 1.0),
            history.float().mean(dim=1),
            (last_gap / (history_size + 1)).clamp(0.0, 1.0),
            torch.zeros(batch_size, device=device),
        ],
        dim=1,
    )
    # The transport target is deliberately about the declared deadline, not
    # the next tick. It is observable from an arrival trace and contains no
    # task, modality, or answer label.
    target = (delay > age.long()) & (delay <= int(deadline))
    return features, target.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=992001)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--deadline", type=float, default=2.0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 8:
        raise ValueError("updates and batch-size are invalid")
    if args.history_size < 1 or args.deadline <= 0:
        raise ValueError("history-size and deadline are invalid")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    predictor = AmodalArrivalPredictor(args.hidden).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.learning_rate)
    curve = []
    start = time.perf_counter()
    predictor.train()
    for update in range(1, args.updates + 1):
        features, target = _trace_batch(
            args.batch_size,
            seed=args.seed + update,
            history_size=args.history_size,
            deadline=args.deadline,
            device=device,
        )
        probability = predictor(features)
        loss = nn.functional.binary_cross_entropy(probability, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append({"update": update, "loss": float(loss.detach())})
    predictor.eval()
    with torch.no_grad():
        test_features, test_target = _trace_batch(
            args.batch_size * 8,
            seed=args.seed + 100_000,
            history_size=args.history_size,
            deadline=args.deadline,
            device=device,
        )
        test_probability = predictor(test_features)
        test_accuracy = float(
            ((test_probability >= 0.5) == test_target.bool()).float().mean()
        )
        test_brier = float((test_probability - test_target).square().mean())
    payload = {
        "schema": "amodal-arrival-predictor-v1",
        "feature_count": AMODAL_WAIT_FEATURES,
        "hidden": args.hidden,
        "state_dict": {
            name: value.detach().cpu() for name, value in predictor.state_dict().items()
        },
        "training": {
            "method": "self-supervised-arrival-before-deadline",
            "labels_used": [],
            "updates": args.updates,
            "batch_size": args.batch_size,
            "history_size": args.history_size,
            "deadline": args.deadline,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    report = {
        "schema": "amodal-arrival-predictor-training-v1",
        "labels_used": [],
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "history_size": args.history_size,
            "deadline": args.deadline,
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "device": str(device),
        },
        "curve": curve,
        "heldout_accuracy": test_accuracy,
        "heldout_brier": test_brier,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"heldout_accuracy": test_accuracy, "heldout_brier": test_brier}))


if __name__ == "__main__":
    main()
