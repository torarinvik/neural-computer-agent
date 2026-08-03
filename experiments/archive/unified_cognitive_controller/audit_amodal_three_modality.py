"""Audit frozen vision+audio+text composition through one amodal bus.

The three frontends were aligned independently to the same neural-IR basis.
This audit loads their saved artifacts without an optimizer, feeds vision plus
audio plus discrete text simultaneously, and checks cross-episode shuffles,
counterfactual reversal, permutation invariance, and N=1 retention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .legacy_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_audio_alignment import (
    PairRelationAudioEncoder,
    render_pair_relation_audio,
)
from .train_amodal_text_alignment import (
    PairRelationTextEncoder,
    render_pair_relation_text_tokens,
)
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(
    controller_path: Path,
    bus_path: Path,
    audio_path: Path,
    text_path: Path,
    device: torch.device,
) -> AmodalControllerRuntime:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    audio_artifact = torch.load(audio_path, map_location="cpu", weights_only=False)
    text_artifact = torch.load(text_path, map_location="cpu", weights_only=False)
    if audio_artifact.get("schema") != "amodal-audio-aligned-frontend-v1":
        raise ValueError("unsupported audio frontend artifact schema")
    if text_artifact.get("schema") != "amodal-text-aligned-frontend-v1":
        raise ValueError("unsupported text frontend artifact schema")
    for artifact, name in ((audio_artifact, "audio"), (text_artifact, "text")):
        if artifact.get("controller_sha256") != _sha256(controller_path):
            raise ValueError(f"{name} frontend targets another controller")
        if artifact.get("input_bus_sha256") != _sha256(bus_path):
            raise ValueError(f"{name} frontend targets another input bus")
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    audio = PairRelationAudioEncoder(
        extracted.controller.width,
        samples=int(audio_artifact["audio_samples"]),
    ).to(device)
    audio.load_state_dict(
        {name: value.to(device=device) for name, value in audio_artifact["state_dict"].items()}
    )
    text = PairRelationTextEncoder(
        extracted.controller.width,
        grid=int(text_artifact["text_grid"]),
        levels=int(text_artifact["text_levels"]),
        embedding_width=int(text_artifact["embedding_width"]),
    ).to(device)
    text.load_state_dict(
        {name: value.to(device=device) for name, value in text_artifact["state_dict"].items()}
    )
    return AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "vision": copy.deepcopy(extracted.encoder),
            "audio": audio,
            "text": text,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()


@torch.no_grad()
def evaluate_three_modality(
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str,
) -> dict[str, float | bool]:
    normal = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, device=device,
    )
    contradictory = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, reverse_contexts=True,
        device=device,
    )
    first, second = split_complementary_views(normal.frames)
    _, contradictory_second = split_complementary_views(contradictory.frames)

    def encode(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = frames.flatten(0, 1)
        audio = render_pair_relation_audio(flat).reshape(count, 6, 1, -1)
        text = render_pair_relation_text_tokens(flat).reshape(count, 6, -1)
        return audio, text

    audio, text = encode(second)
    contradictory_audio, contradictory_text = encode(contradictory_second)

    def run(
        streams: tuple[tuple[torch.Tensor, str], ...],
        *,
        cross_episode: bool = False,
    ) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            named = {
                name: values[:, trial].roll(1, 0) if cross_episode else values[:, trial]
                for values, name in streams
            }
            feedback = torch.full_like(reward, float(trial == 1))
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    vision = run(((first, "vision"),))
    audio_only = run(((audio, "audio"),))
    text_only = run(((text, "text"),))
    vision_audio = run(((first, "vision"), (audio, "audio")))
    vision_text = run(((first, "vision"), (text, "text")))
    triple = run(((first, "vision"), (audio, "audio"), (text, "text")))
    permuted = run(((text, "text"), (first, "vision"), (audio, "audio")))
    shuffled = run(
        ((first, "vision"), (audio, "audio"), (text, "text")),
        cross_episode=True,
    )
    contradictory_triple = run(
        ((first, "vision"), (contradictory_audio, "audio"),
         (contradictory_text, "text"))
    )
    full = run(((normal.frames, "vision"),))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query])
            .float().mean()
        )

    return {
        "vision_only_accuracy": accuracy(vision),
        "audio_only_accuracy": accuracy(audio_only),
        "text_only_accuracy": accuracy(text_only),
        "vision_audio_accuracy": accuracy(vision_audio),
        "vision_text_accuracy": accuracy(vision_text),
        "triple_accuracy": accuracy(triple),
        "shuffled_triple_accuracy": accuracy(shuffled),
        "contradictory_triple_accuracy": accuracy(contradictory_triple),
        "contradictory_prediction_flip_rate": float(
            (triple[:, query] != contradictory_triple[:, query])
            .float().mean()
        ),
        "permuted_action_agreement": float(
            (triple == permuted).float().mean()
        ),
        "full_n1_accuracy": accuracy(full),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=998001)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument(
        "--device",
        default=(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.eval_count < 128:
        raise ValueError("eval-count must provide a meaningful audit")
    device = torch.device(args.device)
    runtime = _load_runtime(
        args.controller, args.input_bus, args.audio, args.text, device
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    results = {
        appearance: evaluate_three_modality(
            runtime, count=args.eval_count, seed=args.seed + index * 10_000,
            device=device, appearance=appearance,
        )
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    thresholds = (0.90, 0.85, 0.90)
    passed = bool(
        controller_unchanged and all(
            row["vision_audio_accuracy"] >= pair_threshold - 0.05
            and row["vision_text_accuracy"] >= pair_threshold
            and row["triple_accuracy"] >= pair_threshold
            and row["vision_only_accuracy"] <= 0.65
            and row["audio_only_accuracy"] <= 0.65
            and row["text_only_accuracy"] <= 0.65
            and row["shuffled_triple_accuracy"] <= 0.60
            and row["contradictory_triple_accuracy"] <= 0.25
            and row["contradictory_prediction_flip_rate"] >= flip_threshold
            and row["permuted_action_agreement"] >= 0.99
            and row["full_n1_accuracy"] >= 0.95
            for row, pair_threshold, flip_threshold in zip(
                results.values(), thresholds, (0.80, 0.70, 0.80)
            )
        )
    )
    report = {
        "schema": "amodal-three-modality-composition-v1",
        "claim": (
            "One frozen controller and input bus compose independently aligned "
            "vision, audio, and discrete text streams."
        ),
        "training_signal": "no_training_in_audit",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "audio": str(args.audio),
        "audio_sha256": _sha256(args.audio),
        "text": str(args.text),
        "text_sha256": _sha256(args.text),
        "configuration": {
            "seed": args.seed, "eval_count": args.eval_count,
            "device": str(device),
            "streams": ["vision", "audio", "text"],
        },
        "results": results,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "results": results}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
