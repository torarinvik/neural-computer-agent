"""Teach the controller to interpret, with no task in the loop.

`controller_pretraining.py` set the precedent: pretrain the controller on
generic mechanics, freeze it, then let external programs carry capability.
This does the same for interpretation.

The skill being taught is fetch-decode-branch, and nothing else:

    given a current event, an instruction, and a workspace summary,
    emit the intention naming the operator that instruction calls for.

An instruction carries two handle fields. When the current event matches the
workspace summary the first field names the operator; otherwise the second
does. Unconditional instructions put the same handle in both, so an
interpreter that ignored the condition would still get those right and fail
every conditional one, which is what separates decoding from branching.

Handles are redrawn at random for every batch. There is no fixed opcode
vocabulary to memorise, so the only thing that generalises is the skill of
reading a field and choosing between two of them — which is why a controller
trained here works on operators invented after it was frozen.

No verifier, no reward, no rendered stimulus, no rule. The training signal is
the machine's own instruction-set semantics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .interpreter_controller import InterpreterController

INTERPRETER_ARTIFACT_SCHEMA = "neural-computer.interpreter-controller.v1"
MATCH_TOLERANCE = 0.5


def sample_batch(
    *,
    batch: int,
    event_width: int,
    operators: int,
    generator: torch.Generator,
    tolerance: float = MATCH_TOLERANCE,
) -> dict[str, torch.Tensor]:
    """One batch of interpretation problems with freshly drawn handles."""

    handles = torch.randn(batch, operators, event_width, generator=generator)
    handles = handles / handles.norm(dim=2, keepdim=True)
    primary = torch.randint(0, operators, (batch,), generator=generator)
    alternate = torch.randint(0, operators, (batch,), generator=generator)
    workspace = torch.randn(batch, event_width, generator=generator)
    # Half the batch matches the workspace, half does not.
    matched = torch.rand(batch, generator=generator) < 0.5
    near = workspace + 0.05 * torch.randn(batch, event_width, generator=generator)
    far = workspace + 4.0 * torch.randn(batch, event_width, generator=generator)
    event = torch.where(matched[:, None], near, far)
    distance = torch.linalg.vector_norm(event - workspace, dim=1)
    condition = distance <= tolerance
    rows = torch.arange(batch)
    instruction = torch.cat(
        (handles[rows, primary], handles[rows, alternate]), dim=1
    )
    target_index = torch.where(condition, primary, alternate)
    return {
        "event": event,
        "instruction": instruction,
        "workspace": workspace,
        "handles": handles,
        "target_index": target_index,
        "target": handles[rows, target_index],
        "condition": condition,
    }


def interpretation_accuracy(
    controller: InterpreterController, batch: dict[str, torch.Tensor]
) -> float:
    """Share of problems whose emitted intention resolves to the right operator."""

    with torch.no_grad():
        intention = controller(
            batch["event"], batch["instruction"], batch["workspace"]
        )
        scores = torch.einsum("bow,bw->bo", batch["handles"], intention)
        chosen = scores.argmax(dim=1)
    return float((chosen == batch["target_index"]).float().mean())


def pretrain_interpreter(
    *,
    event_width: int,
    operators: int = 8,
    steps: int = 4000,
    batch: int = 256,
    learning_rate: float = 3e-3,
    seed: int = 1001,
    hidden: int = 64,
) -> tuple[InterpreterController, dict[str, Any]]:
    """Train, then freeze. Returns the controller and its training record."""

    torch.manual_seed(int(seed))
    controller = InterpreterController(event_width, hidden=hidden)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed) + 1)
    started = time.perf_counter()
    curve: list[dict[str, float]] = []
    for step in range(steps):
        problems = sample_batch(
            batch=batch,
            event_width=event_width,
            operators=operators,
            generator=generator,
        )
        intention = controller(
            problems["event"], problems["instruction"], problems["workspace"]
        )
        # Pull the intention onto the right handle and off the wrong ones.
        scores = torch.einsum("bow,bw->bo", problems["handles"], intention)
        loss = torch.nn.functional.cross_entropy(
            scores * 8.0, problems["target_index"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach())})
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    controller.eval()
    record = {
        "schema": INTERPRETER_ARTIFACT_SCHEMA,
        "event_width": int(event_width),
        "operators_seen_in_training": int(operators),
        "steps": int(steps),
        "batch": int(batch),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "hidden": int(hidden),
        "loss_curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "controller_digest": controller.digest(),
        "parameters": sum(p.numel() for p in controller.parameters()),
    }
    return controller, record


def evaluate_generalisation(
    controller: InterpreterController,
    *,
    event_width: int,
    seed: int,
    batch: int = 4096,
    operator_counts: tuple[int, ...] = (2, 4, 8, 16, 32),
) -> dict[str, Any]:
    """Held-out handles, and vocabularies the controller never trained on."""

    generator = torch.Generator().manual_seed(int(seed))
    rows: list[dict[str, Any]] = []
    for operators in operator_counts:
        problems = sample_batch(
            batch=batch,
            event_width=event_width,
            operators=operators,
            generator=generator,
        )
        overall = interpretation_accuracy(controller, problems)
        met = problems["condition"]
        conditional = {
            "condition_met": float(
                (
                    torch.einsum(
                        "bow,bw->bo",
                        problems["handles"][met],
                        controller(
                            problems["event"][met],
                            problems["instruction"][met],
                            problems["workspace"][met],
                        ),
                    ).argmax(dim=1)
                    == problems["target_index"][met]
                )
                .float()
                .mean()
            ),
            "condition_unmet": float(
                (
                    torch.einsum(
                        "bow,bw->bo",
                        problems["handles"][~met],
                        controller(
                            problems["event"][~met],
                            problems["instruction"][~met],
                            problems["workspace"][~met],
                        ),
                    ).argmax(dim=1)
                    == problems["target_index"][~met]
                )
                .float()
                .mean()
            ),
        }
        rows.append(
            {
                "operators": operators,
                "accuracy": overall,
                **conditional,
                "chance": 1.0 / operators,
            }
        )
    return {"held_out_handles": rows}


def save_interpreter(controller: InterpreterController, record: dict[str, Any], path: Path) -> str:
    payload = {
        "schema": INTERPRETER_ARTIFACT_SCHEMA,
        "record": record,
        "state": controller.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def load_interpreter(path: Path) -> tuple[InterpreterController, dict[str, Any]]:
    payload = torch.load(path, weights_only=False)
    if payload.get("schema") != INTERPRETER_ARTIFACT_SCHEMA:
        raise ValueError("unsupported interpreter controller artifact")
    record = payload["record"]
    controller = InterpreterController(
        int(record["event_width"]), hidden=int(record["hidden"])
    )
    controller.load_state_dict(payload["state"])
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    controller.eval()
    return controller, record


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--operators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=repository / "artifacts/checkpoints/interpreter_controller_seed1001.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "session_records" / "brainworkshop_interpreter_pretraining_2026-08-15",
    )
    arguments = parser.parse_args()
    controller, record = pretrain_interpreter(
        event_width=arguments.event_width,
        operators=arguments.operators,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    generalisation = evaluate_generalisation(
        controller, event_width=arguments.event_width, seed=arguments.seed + 99
    )
    digest = save_interpreter(controller, record, arguments.artifact)
    report = {**record, **generalisation, "artifact_sha256": digest}
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "pretraining.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (arguments.output_dir / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(arguments.output_dir.glob("*.json"))
        )
        + "\n"
    )
    print(json.dumps({"held_out_handles": generalisation["held_out_handles"],
                      "parameters": record["parameters"],
                      "wall_seconds": record["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
