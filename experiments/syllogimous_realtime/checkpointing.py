"""Versioned adapter checkpoint export/import."""
from __future__ import annotations
from pathlib import Path
import json
import torch

def save_adapter(path: str | Path, module: torch.nn.Module, *, config: dict,
                 metrics: dict, seed: int, compiler: str = "~/.elisac/elisac") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "syllogimous.adapter-checkpoint.v1",
                "state_dict": module.state_dict(), "config": config,
                "metrics": metrics, "seed": seed, "compiler": compiler}, path)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps({
        "schema": "syllogimous.adapter-checkpoint.v1", "config": config,
        "metrics": metrics, "seed": seed, "compiler": compiler,
    }, indent=2, sort_keys=True) + "\n")

def load_adapter(path: str | Path, module: torch.nn.Module) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "syllogimous.adapter-checkpoint.v1":
        raise ValueError("unsupported adapter checkpoint schema")
    module.load_state_dict(payload["state_dict"])
    return {key: payload[key] for key in ("config", "metrics", "seed", "compiler")}
