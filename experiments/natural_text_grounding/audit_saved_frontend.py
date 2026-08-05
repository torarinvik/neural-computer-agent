"""Replay a saved grounded-text frontend without its training optimizer.

This audit intentionally reloads only the frozen controller/input bus and the
serialized frontend state.  It is the independent saved-artifact check used
before a frontend can be promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.natural_text_grounding.external_caption_source import (
    ANNOTATION_TABLE_V3_PATH,
    CORPUS_V2_PATH,
    corpus_sha256,
    render_external_annotation_text_v3,
    render_external_text,
    render_external_text_v2,
)
from experiments.natural_text_grounding.train import (
    ALL_STYLES,
    HELDOUT_STYLES,
    _sha256,
    evaluate,
    load_runtime,
    render_grounded_text,
)

APPEARANCES = ("bars", "diamonds", "dot_pairs")
STRICT_GATE = {
    "fused_accuracy_minimum": 0.90,
    "shuffled_partner_accuracy_maximum": 0.60,
    "contradictory_partner_accuracy_maximum": 0.25,
    "contradictory_prediction_flip_rate_minimum": 0.75,
    "vision_only_accuracy_minimum": 0.95,
}


def _frontend_kind(payload: dict[str, object]) -> str:
    kind = str(payload.get("frontend_kind", ""))
    if kind.endswith(("_cnn", "_relative_order")):
        return "cnn"
    if kind.endswith("_transformer"):
        return "transformer"
    raise ValueError(f"unsupported saved frontend kind: {kind!r}")


def _passes(rows: list[dict[str, float]], controller_unchanged: bool) -> bool:
    return bool(
        controller_unchanged
        and all(
            row["fused_accuracy"] >= STRICT_GATE["fused_accuracy_minimum"]
            and row["shuffled_partner_accuracy"]
            <= STRICT_GATE["shuffled_partner_accuracy_maximum"]
            and row["contradictory_partner_accuracy"]
            <= STRICT_GATE["contradictory_partner_accuracy_maximum"]
            and row["contradictory_prediction_flip_rate"]
            >= STRICT_GATE["contradictory_prediction_flip_rate_minimum"]
            and row["full_n1_accuracy"] >= STRICT_GATE["vision_only_accuracy_minimum"]
            for row in rows
        )
    )


def audit_saved_frontend(
    *,
    controller_path: Path,
    bus_path: Path,
    adapter_path: Path,
    report_path: Path,
    seed: int,
    count: int,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(adapter_path, map_location=device, weights_only=False)
    if payload.get("schema") != "amodal-grounded-byte-text-frontend-v1":
        raise ValueError("adapter does not use the grounded-text frontend schema")
    controller_sha256 = _sha256(controller_path)
    input_bus_sha256 = _sha256(bus_path)
    if payload.get("controller_sha256") != controller_sha256:
        raise ValueError("adapter controller hash does not match audit input")
    if payload.get("input_bus_sha256") != input_bus_sha256:
        raise ValueError("adapter input-bus hash does not match audit input")

    runtime, frontend = load_runtime(
        controller_path,
        bus_path,
        device,
        position_bins=int(payload.get("position_bins", 0)),
        frontend_kind=_frontend_kind(payload),
    )
    frontend.load_state_dict(payload["state_dict"])
    runtime.eval()
    frontend.eval()
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    final_by_appearance: dict[str, dict[str, dict[str, float]]] = {}
    text_source = str(payload.get("text_source", "pixel_template"))
    if text_source == "external_corpus":
        text_renderer = render_external_text
        if payload.get("caption_corpus_sha256") != corpus_sha256():
            raise ValueError("saved frontend caption corpus does not match audit input")
    elif text_source == "external_corpus_v2":
        text_renderer = render_external_text_v2
        if payload.get("caption_corpus_sha256") != corpus_sha256(CORPUS_V2_PATH):
            raise ValueError("saved frontend v2 caption corpus does not match audit input")
    elif text_source == "external_annotation_table_v3":
        text_renderer = render_external_annotation_text_v3
        if payload.get("caption_corpus_sha256") != corpus_sha256(
            ANNOTATION_TABLE_V3_PATH
        ):
            raise ValueError(
                "saved frontend v3 annotation table does not match audit input"
            )
    elif text_source == "pixel_template":
        text_renderer = render_grounded_text
    else:
        raise ValueError(f"unsupported saved frontend text source: {text_source!r}")
    for appearance_index, appearance in enumerate(APPEARANCES):
        final_by_appearance[appearance] = {}
        for style in ALL_STYLES:
            final_by_appearance[appearance][str(style)] = evaluate(
                runtime,
                count=count,
                seed=seed + 60_000 + appearance_index * 10_000 + style * 100,
                device=device,
                appearance=appearance,
                style=style,
                text_renderer=text_renderer,
            )
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    heldout_rows = [
        final_by_appearance[appearance][str(style)]
        for appearance in APPEARANCES
        for style in HELDOUT_STYLES
    ]
    report = {
        "schema": "amodal-grounded-byte-text-replay-audit-v2",
        "adapter": str(adapter_path),
        "adapter_sha256": _sha256(adapter_path),
        "controller": str(controller_path),
        "controller_sha256": controller_sha256,
        "input_bus": str(bus_path),
        "input_bus_sha256": input_bus_sha256,
        "frontend_kind": payload["frontend_kind"],
        "text_source": text_source,
        "caption_corpus_sha256": payload.get("caption_corpus_sha256"),
        "training_optimizer_loaded": False,
        "evaluation_count_per_appearance_style": count,
        "seed": seed,
        "heldout_styles": HELDOUT_STYLES,
        "strict_gate": STRICT_GATE,
        "final_by_appearance": final_by_appearance,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": _passes(heldout_rows, controller_unchanged),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1001001)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()
    if args.count < 64:
        raise ValueError("count must be at least 64 for the saved-artifact audit")
    report = audit_saved_frontend(
        controller_path=args.controller,
        bus_path=args.input_bus,
        adapter_path=args.adapter,
        report_path=args.report,
        seed=args.seed,
        count=args.count,
        device=torch.device(args.device),
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
