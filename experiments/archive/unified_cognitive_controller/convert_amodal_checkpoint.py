"""Convert a legacy unified checkpoint into three extracted components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .legacy_runtime import (
    convert_legacy_checkpoint,
    runtime_from_extracted_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_payload = torch.load(
        args.source, map_location=args.device, weights_only=False
    )
    convert_legacy_checkpoint(args.source, args.destination, device=args.device)
    extracted_payload = torch.load(
        args.destination, map_location=args.device, weights_only=False
    )
    runtime = runtime_from_extracted_payload(extracted_payload, device=args.device)
    rebuilt = runtime.legacy_state_dict()
    exact = set(rebuilt) == set(source_payload["state_dict"]) and all(
        torch.equal(value, rebuilt[name])
        for name, value in source_payload["state_dict"].items()
    )
    if not exact:
        raise RuntimeError("converted components do not reconstruct source weights")
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "format": extracted_payload["format"],
                "event_schema": extracted_payload["event_schema"],
                "intention_schema": extracted_payload["intention_schema"],
                "source_state_tensors": len(source_payload["state_dict"]),
                "reconstructed_exactly": exact,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
