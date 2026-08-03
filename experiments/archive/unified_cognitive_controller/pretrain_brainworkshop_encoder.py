"""Self-supervised frontend pretraining for the Brain Workshop gym.

No stimulus identity, match flag, action, or verifier target is used here.
The selected encoder is trained only to reconstruct its observed vision frame
or audio waveform through a throwaway decoder. The decoder is discarded
before any behavioral probe; the surviving artifact is a modality-specific
frontend that preserves the event stream without semantic labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .brainworkshop_gym import (
    BrainWorkshopAudioEncoder, BrainWorkshopConfig, BrainWorkshopVisionEncoder)


class FrameReconstructionDecoder(nn.Module):
    """Disposable pixel decoder; it is never connected to the controller."""

    def __init__(self, event_width: int, output_size: int = 60) -> None:
        super().__init__()
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(event_width, 256), nn.GELU(),
            nn.Linear(256, 3 * output_size * output_size),
        )

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        return self.network(event).view(
            event.shape[0], 3, self.output_size, self.output_size).sigmoid()


class AudioReconstructionDecoder(nn.Module):
    """Disposable waveform decoder; it is never connected to the controller."""

    def __init__(self, event_width: int, samples: int = 800) -> None:
        super().__init__()
        self.samples = samples
        self.network = nn.Sequential(
            nn.Linear(event_width, 256), nn.GELU(),
            nn.Linear(256, samples),
        )

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        return self.network(event).view(event.shape[0], 1, self.samples).tanh()


def _device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def _observations(config: BrainWorkshopConfig, *, count: int, seed: int,
                  device: torch.device, modality: str) -> torch.Tensor:
    # The match schedule is irrelevant to reconstruction, but using the gym's
    # public observations keeps this probe tied to the exact sensory stream.
    observations = []
    for index in range(count):
        episode_seed = seed + index
        from .brainworkshop_gym import generate_brainworkshop_episode

        episode = generate_brainworkshop_episode(
            config, seed=episode_seed, device=device)
        values = [
            observation.vision if modality == "vision" else observation.audio
            for observation in episode.observations]
        observations.append(torch.stack(values))
    return torch.cat(observations, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=45101)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--event-width", type=int, default=32)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--position-vocab", type=int, default=2,
                        choices=(2, 4, 8))
    parser.add_argument("--modality", choices=("vision", "audio"),
                        default="vision")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 4:
        raise ValueError("updates and batch size must be positive")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    config = BrainWorkshopConfig(
        n_back=1, trials=args.trials, position_vocab=args.position_vocab,
        modalities=(args.modality,), balanced_matches=False)
    encoder = (
        BrainWorkshopVisionEncoder(args.event_width)
        if args.modality == "vision" else
        BrainWorkshopAudioEncoder(args.event_width)
    ).to(device)
    decoder = (
        FrameReconstructionDecoder(args.event_width)
        if args.modality == "vision" else
        AudioReconstructionDecoder(
            args.event_width, samples=config.audio_samples)
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        encoder.train()
        decoder.train()
        observations = _observations(
            config, count=args.batch_size, seed=args.seed + update * 1000,
            device=device, modality=args.modality)
        target = (
            F.interpolate(
                observations, size=(60, 60), mode="bilinear",
                align_corners=False)
            if args.modality == "vision" else observations)
        event = encoder(observations)
        reconstruction = decoder(event)
        loss = F.mse_loss(reconstruction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0))
        optimizer.step()
        record = {
            "update": update,
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "unique_observations": int(observations.shape[0]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        if update == 1 or update == args.updates or update % 8 == 0:
            print(json.dumps(record, sort_keys=True), flush=True)

    encoder.eval()
    report = {
        "experiment": (
            "brainworkshop_self_supervised_frame_reconstruction"
            if args.modality == "vision" else
            "brainworkshop_self_supervised_audio_reconstruction"),
        "objective": (
            "pixel_reconstruction_only" if args.modality == "vision"
            else "waveform_reconstruction_only"),
        "modality": args.modality,
        "device": str(device),
        "config": config.__dict__,
        "event_width": args.event_width,
        "updates": args.updates,
        "unique_frames": args.updates * args.batch_size * args.trials,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gate": {
            "gradient_alive": all(
                record["gradient_norm"] > 1e-8 for record in history),
            "reconstruction_improved": (
                history[-1]["loss"] < history[0]["loss"]),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save({
        "format": "brainworkshop-self-supervised-encoder.v1",
        "event_width": args.event_width,
        "config": config.__dict__,
        "encoder_state_dict": encoder.state_dict(),
        "report": str(args.report),
    }, args.checkpoint_out)
    print(json.dumps({
        "gate": report["gate"], "final_loss": history[-1]["loss"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
