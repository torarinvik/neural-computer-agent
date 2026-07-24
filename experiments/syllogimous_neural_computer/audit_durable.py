from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.syllogimous_latent_agent.data import collate_episodes

from .lifetime import generate_sensory_lifetime
from .memory import PersistentMemory
from .model import NeuralComputerAgent


def intervene(memory: PersistentMemory, mode: str) -> PersistentMemory:
    if mode == "intact":
        return memory
    if mode == "empty":
        return PersistentMemory.empty(1, memory.width, device=memory.keys.device)
    changed = memory.clone()
    indices = changed.valid.nonzero(as_tuple=False).squeeze(1)
    if mode == "shuffled":
        if indices.numel() > 1:
            changed.values[indices] = changed.values[indices.roll(1)].clone()
        else:
            changed.values[indices] = -changed.values[indices]
    elif mode == "garbage":
        key_index = torch.arange(changed.keys[indices].numel(), device=changed.keys.device,
                                 dtype=changed.keys.dtype).reshape_as(changed.keys[indices])
        changed.keys[indices] = torch.sin(key_index * 1.618)
        changed.values[indices] = torch.cos((key_index + 0.5) * 2.414)
    else:
        raise ValueError(f"unknown intervention {mode!r}")
    return changed


@torch.no_grad()
def audit(model: NeuralComputerAgent, *, device: torch.device, samples: int,
          associations: int, delay: int, choices: int, threshold: float,
          blob: Path, intervention: str) -> dict[str, float]:
    model.eval()
    correct = queries = writes = 0
    for sample in range(samples):
        lifetime = generate_sensory_lifetime(2_000_000 + sample,
                                             associations=associations,
                                             delay=delay, choices=choices, heldout=True)
        memory = PersistentMemory.empty(4, model.hidden, device=device, growth_chunk=4)
        query_start = associations + delay
        for step, episode in enumerate(lifetime.episodes):
            if step == query_start:
                memory.save(blob)
                memory = PersistentMemory.load(blob, device=device)
                memory = intervene(memory, intervention)
            batch = collate_episodes([episode])
            output = model(batch["frames"].to(device), batch["pcm"].to(device),
                           batch["mask"].to(device), memory)
            if step >= query_start:
                prediction = int(output.answer_logits[0, -1].argmax())
                correct += prediction == int(batch["actions"][0, 0])
                queries += 1
            else:
                admitted = output.write_strengths >= threshold
                if admitted.any():
                    writes += memory.write(output.write_keys[admitted],
                                           output.write_values[admitted],
                                           output.write_strengths[admitted], threshold=0.0)
    return {"accuracy": correct / max(1, queries),
            "writes_per_lifetime": writes / samples,
            "samples": float(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart/reload audit of discrete neural memory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--associations", type=int, default=1)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload["arguments"]
    model = NeuralComputerAgent(config["hidden"], config["workspace_slots"], config["heads"],
                                config["thought_steps"], config["choices"]).to(args.device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
        raise ValueError(f"incompatible checkpoint: {incompatible}")
    results = {
        mode: audit(model, device=torch.device(args.device), samples=args.samples,
                    associations=args.associations, delay=args.delay, choices=args.choices,
                    threshold=args.threshold, blob=args.blob, intervention=mode)
        for mode in ("intact", "empty", "shuffled", "garbage")
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "schema": "syllogimous-durable-memory-audit-v1",
        "checkpoint": str(args.checkpoint), "threshold": args.threshold,
        "process_boundary": "save-and-reload-before-query", "results": results,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
