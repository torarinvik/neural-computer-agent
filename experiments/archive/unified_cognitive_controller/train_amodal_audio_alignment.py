"""Align a raw audio frontend to the frozen vision neural-IR basis.

The environment renders a deterministic waveform from each ordinary visual
view. The waveform is a separate sensor stream; it contains no task label,
identity, relation bit, or correct action. During alignment, the audio encoder
is trained only to agree with the frozen vision encoder on paired views. The
controller, input bus, vision frontend, and decoder stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .legacy_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_pair_relation_audio(
    frames: torch.Tensor,
    *,
    samples: int = 2_048,
) -> torch.Tensor:
    """Render a view as a waveform using fixed sensor physics.

    The sensor preserves a low-resolution RGB image as amplitudes of distinct
    carrier frequencies. This is deliberately a modality conversion, not a
    semantic shortcut: the learner sees only the waveform and never receives
    the pooled pixels or the generator's private relation.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [batch, 3, height, width]")
    pooled = F.adaptive_avg_pool2d(frames, (16, 16)).flatten(1)
    pooled = pooled - pooled.mean(dim=-1, keepdim=True)
    pooled = pooled / pooled.std(dim=-1, keepdim=True).clamp_min(1e-4)
    feature_count = pooled.shape[-1]
    if samples < feature_count * 2:
        raise ValueError("audio sample count is too short for the sensor grid")
    time = torch.arange(samples, device=frames.device, dtype=frames.dtype)
    # Place each feature on an exact FFT bin. This is ordinary fixed spectral
    # front-end physics and avoids making the learned encoder invert severe
    # carrier leakage before it can learn the neural-IR projection.
    frequencies = torch.arange(
        1, feature_count + 1, device=frames.device, dtype=frames.dtype
    )
    carriers = torch.sin(
        2.0 * math.pi * frequencies[:, None] * time[None, :] / samples
    )
    waveform = torch.einsum("bf,fs->bs", pooled, carriers)
    return waveform.unsqueeze(1)


class PairRelationAudioEncoder(nn.Module):
    """Small replaceable spectral frontend with no task-specific branch."""

    def __init__(self, event_width: int, *, samples: int = 2_048) -> None:
        super().__init__()
        self.samples = samples
        self.feature_count = 3 * 16 * 16
        self.head = nn.Sequential(
            nn.Linear(self.feature_count, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, event_width),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        if waveform.shape[-1] != self.samples:
            raise ValueError("waveform length does not match the frontend")
        spectrum = torch.fft.rfft(waveform[:, 0], dim=-1)
        if spectrum.shape[-1] <= self.feature_count:
            raise ValueError("audio sample count is too short for the sensor grid")
        # The renderer uses sine carriers. Demodulation is fixed sensor DSP;
        # the learned head still has to map recovered raw audio content into
        # the controller's opaque neural-IR basis.
        recovered = (-2.0 / self.samples) * spectrum.imag[
            :, 1 : self.feature_count + 1
        ]
        return self.head(recovered)


def _load_runtime(
    controller_path: Path,
    bus_path: Path,
    device: torch.device,
    *,
    audio_samples: int,
) -> tuple[AmodalControllerRuntime, PairRelationAudioEncoder]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    audio = PairRelationAudioEncoder(
        extracted.controller.width, samples=audio_samples
    ).to(device)
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": audio,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in audio.parameters():
        parameter.requires_grad_(True)
    return runtime, audio


@torch.no_grad()
def evaluate_audio_alignment(
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str,
    audio_samples: int,
) -> dict[str, float]:
    normal = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        device=device,
    )
    contradictory = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        reverse_contexts=True,
        device=device,
    )
    first, second = split_complementary_views(normal.frames)
    _, contradictory_second = split_complementary_views(contradictory.frames)
    first_audio = render_pair_relation_audio(
        first.flatten(0, 1), samples=audio_samples
    ).reshape(
        count, normal.trials, 1, -1
    )
    second_audio = render_pair_relation_audio(
        second.flatten(0, 1), samples=audio_samples
    ).reshape(
        count, normal.trials, 1, -1
    )
    contradictory_audio = render_pair_relation_audio(
        contradictory_second.flatten(0, 1), samples=audio_samples
    ).reshape(count, normal.trials, 1, -1)

    def run(streams: tuple[tuple[torch.Tensor, str], ...]) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            feedback = torch.full_like(reward, float(trial == 1))
            named = {
                name: values[:, trial]
                for values, name in streams
            }
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    first_actions = run(((first, "stream_a"),))
    second_actions = run(((second_audio, "stream_b"),))
    fused_actions = run(((first, "stream_a"), (second_audio, "stream_b")))
    shuffled_actions = run(
        ((first, "stream_a"), (second_audio.roll(1, 0), "stream_b"))
    )
    contradictory_actions = run(
        ((first, "stream_a"), (contradictory_audio, "stream_b"))
    )
    full_actions = run(((normal.frames, "stream_a"),))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query])
            .float()
            .mean()
        )

    return {
        "stream_a_accuracy": accuracy(first_actions),
        "stream_b_accuracy": accuracy(second_actions),
        "fused_accuracy": accuracy(fused_actions),
        "shuffled_partner_accuracy": accuracy(shuffled_actions),
        "contradictory_partner_accuracy": accuracy(contradictory_actions),
        "contradictory_prediction_flip_rate": float(
            (fused_actions[:, query] != contradictory_actions[:, query])
            .float()
            .mean()
        ),
        "full_n1_accuracy": accuracy(full_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=289001)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--audio-samples", type=int, default=2_048)
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
    if args.updates < 1 or args.batch_size < 4 or args.eval_count < 64:
        raise ValueError("updates, batch-size, and eval-count are too small")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, audio = _load_runtime(
        args.controller,
        args.input_bus,
        device,
        audio_samples=args.audio_samples,
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    optimizer = torch.optim.Adam(audio.parameters(), lr=args.learning_rate)
    appearances = ("bars", "diamonds", "dot_pairs")
    initial_by_appearance = {
        appearance: evaluate_audio_alignment(
            runtime,
            count=args.eval_count,
            seed=args.seed + index * 10_000,
            device=device,
            appearance=appearance,
            audio_samples=args.audio_samples,
        )
        for index, appearance in enumerate(appearances)
    }
    curve = [{"update": 0, "paired_frames": 0, **initial_by_appearance["bars"]}]
    paired_frames = 0
    for update in range(1, args.updates + 1):
        runtime.train()
        losses = []
        for appearance_index, appearance in enumerate(appearances):
            batch = generate_lifetimes(
                args.batch_size,
                6,
                seed=args.seed + update * 100 + appearance_index,
                heldout=True,
                task="pair_relation",
                appearance=appearance,
                support_trials=1,
                device=device,
            )
            first, second = split_complementary_views(batch.frames)
            views = torch.cat((first.flatten(0, 1), second.flatten(0, 1)), dim=0)
            waveforms = render_pair_relation_audio(
                views, samples=args.audio_samples
            )
            with torch.no_grad():
                target = runtime.encoders["stream_a"](views)
            prediction = audio(waveforms)
            losses.append(F.mse_loss(prediction, target))
            paired_frames += int(views.shape[0])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(audio.parameters(), 5.0)
        optimizer.step()
        runtime.eval()
        curve.append(
            {
                "update": update,
                "paired_frames": paired_frames,
                "loss": float(loss.detach()),
                **evaluate_audio_alignment(
                    runtime,
                    count=args.eval_count,
                    seed=args.seed,
                    device=device,
                    appearance="bars",
                    audio_samples=args.audio_samples,
                ),
            }
        )
    final_by_appearance = {
        appearance: evaluate_audio_alignment(
            runtime,
            count=args.eval_count * 2,
            seed=args.seed + 60_000 + index * 10_000,
            device=device,
            appearance=appearance,
            audio_samples=args.audio_samples,
        )
        for index, appearance in enumerate(appearances)
    }
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    passed = controller_unchanged and all(
        row["fused_accuracy"] >= threshold
        and row["stream_a_accuracy"] <= 0.65
        and row["stream_b_accuracy"] <= 0.65
        and row["shuffled_partner_accuracy"] <= 0.60
        and row["contradictory_partner_accuracy"] <= 0.25
        and row["contradictory_prediction_flip_rate"] >= flip_threshold
        and row["full_n1_accuracy"] >= 0.95
        for row, threshold, flip_threshold in zip(
            final_by_appearance.values(), (0.85, 0.80, 0.85), (0.70, 0.65, 0.70)
        )
    )
    report = {
        "schema": "amodal-audio-basis-alignment-v1",
        "claim": (
            "A raw audio frontend can align to a frozen vision neural-IR basis "
            "using paired sensory consistency without semantic labels."
        ),
        "training_signal": "paired_raw_audio_vision_consistency",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "eval_count": args.eval_count,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "audio_samples": args.audio_samples,
            "paired_views_per_update": 2,
            "appearances": appearances,
            "trainable_components": ["audio_encoder"],
        },
        "initial_by_appearance": initial_by_appearance,
        "curve": curve,
        "final_by_appearance": final_by_appearance,
        "paired_frames": paired_frames,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "schema": "amodal-audio-aligned-frontend-v1",
            "controller_sha256": _sha256(args.controller),
            "input_bus_sha256": _sha256(args.input_bus),
            "latent_width": runtime.controller.width,
            "frontend_kind": "pair_relation_audio_waveform",
            "audio_samples": args.audio_samples,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in audio.state_dict().items()
            },
            "training_signal": report["training_signal"],
            "source_report": str(args.report),
            "passed": passed,
        },
        args.adapter_out,
    )
    print(json.dumps({"passed": passed, "final": final_by_appearance}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
