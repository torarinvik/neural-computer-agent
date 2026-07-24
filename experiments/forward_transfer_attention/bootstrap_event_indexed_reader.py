"""Install an audited event-indexed reader behind an exact-zero residual."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--reader-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.0)
    args = parser.parse_args()
    payload = torch.load(
        args.controller_checkpoint, map_location="cpu", weights_only=False)
    reader = torch.load(
        args.reader_checkpoint, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    for key, value in reader["model"].items():
        state["event_indexed_memory_reader." + key] = value
    state["event_indexed_memory_reader_strength"] = torch.tensor(args.strength)
    config_key = ("controller_arguments" if "controller_arguments" in payload
                  else "arguments")
    config = dict(payload[config_key])
    if int(reader["hidden"]) != int(config["hidden"]):
        raise ValueError("reader/controller hidden widths do not match")
    config["event_indexed_memory_reader"] = True
    config["event_indexed_memory_reader_width"] = int(reader["width"])
    config["event_indexed_memory_reader_architecture"] = reader.get(
        "architecture", "relation")
    payload["model"] = state
    payload[config_key] = config
    payload["event_indexed_reader_bootstrap"] = {
        "source_controller": str(args.controller_checkpoint),
        "source_controller_sha256": _sha256(args.controller_checkpoint),
        "source_reader": str(args.reader_checkpoint),
        "source_reader_sha256": _sha256(args.reader_checkpoint),
        "strength": args.strength,
        "supervised": True,
        "sensory_only": True,
        "event_order_invariant": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
