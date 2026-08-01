"""Remove the legacy action suffix by folding it into base intention space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .amodal_runtime import (
    canonicalize_action_adapter_checkpoint,
    runtime_from_legacy_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    canonicalize_action_adapter_checkpoint(
        args.source, args.destination, device=args.device
    )
    payload = torch.load(args.destination, map_location=args.device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=args.device)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "migration": payload["migration"],
                "controller_owns_encoder": runtime.controller.vision is not None,
                "controller_owns_decoder": runtime.controller.actuator is not None,
                "compatibility_suffix_is_structurally_zero": (
                    not runtime.compatibility_suffix_active
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
