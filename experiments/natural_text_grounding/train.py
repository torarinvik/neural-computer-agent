"""Align raw byte text to a frozen amodal neural-IR controller.

The text source describes only visible properties of the rendered scene.  The
trainable frontend receives padded UTF-8 bytes, not descriptor IDs.  Training
uses paired consistency with a frozen visual encoder; no reward, correct
action, task ID, or semantic target enters the learner.  This is a stronger
surface-language bridge than the archived fixed word-ID caption adapter, but
it remains synthetic grounded language rather than a pretrained language
model or open-world language understanding claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from experiments.archive.unified_cognitive_controller.environment import (
    NULL_ACTION,
    generate_lifetimes,
)
from experiments.archive.unified_cognitive_controller.legacy_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from experiments.archive.unified_cognitive_controller.train_complementary_input_bus import (
    split_complementary_views,
)

from .external_caption_source import (
    ANNOTATION_TABLE_V3_PATH,
    CORPUS_V2_PATH,
    corpus_sha256,
    render_external_annotation_text_v3,
    render_external_text,
    render_external_text_v2,
)

TEXT_LENGTH = 128
TEXT_VOCAB = 257  # zero is padding; byte values are shifted by one
TRAIN_STYLES = (0, 1, 2)
HELDOUT_STYLES = (3, 4)
ALL_STYLES = TRAIN_STYLES + HELDOUT_STYLES
TextRenderer = Callable[..., torch.Tensor]


def _descriptors(
    frames: torch.Tensor,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Extract observable textual descriptors from pixels for the captioner."""
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [batch, 3, height, width]")
    max_value = frames.max(dim=1).values
    min_value = frames.min(dim=1).values
    visible = (max_value - min_value > 0.12) & (max_value > 0.15)
    height, width = frames.shape[-2:]
    ys = torch.arange(height, device=frames.device).view(1, height, 1)
    xs = torch.arange(width, device=frames.device).view(1, 1, width)
    # The text source is fed a complementary pixel view. Prefer the visible
    # object on that view's side of the neutral diagonal; fall back to all
    # visible pixels for arbitrary caller-provided images. No verifier ID or
    # renderer-side object identity enters this caption.
    diagonal_visible = visible & (xs < ys)
    diagonal_mass = diagonal_visible.float().sum((1, 2))
    use_diagonal = diagonal_mass >= 8.0
    selected = torch.where(use_diagonal[:, None, None], diagonal_visible, visible)
    mass = selected.float().sum((1, 2)).clamp_min(1.0)
    rgb = (frames * selected.unsqueeze(1)).sum((2, 3)) / mass.unsqueeze(1)
    colours = ["red", "green", "blue"]
    colour = [colours[int(value)] for value in rgb.argmax(dim=-1)]
    brightness = [
        "bright" if bool(value) else "dim"
        for value in (rgb.mean(dim=-1) > 0.45)
    ]
    centre_y = (selected * ys).sum((1, 2)) / mass
    centre_x = (selected * xs).sum((1, 2)) / mass
    horizontal = ["left" if value <= width / 2 else "right" for value in centre_x]
    vertical = ["upper" if value <= height / 2 else "lower" for value in centre_y]
    variance_y = (
        selected * (ys - centre_y[:, None, None]).square()
    ).sum((1, 2)) / mass
    variance_x = (
        selected * (xs - centre_x[:, None, None]).square()
    ).sum((1, 2)) / mass
    tall = variance_y.sqrt() >= variance_x.sqrt()
    shapes = ["tall form" if value else "wide form" for value in tall]
    return colour, brightness, horizontal, vertical, shapes


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def render_grounded_text(
    frames: torch.Tensor,
    *,
    style: int | Sequence[int] = 0,
    text_length: int = TEXT_LENGTH,
) -> torch.Tensor:
    """Render variable natural-language descriptions as shifted byte tokens.

    Styles 0--2 are training paraphrases.  Styles 3--4 are held-out syntax and
    word-order variants.  The external renderer sees pixels and visible object
    descriptors only; the returned bytes contain no hidden verifier metadata.
    """
    if text_length < 32:
        raise ValueError("text_length is too small for grounded descriptions")
    if isinstance(style, int):
        styles = [style] * frames.shape[0]
    else:
        styles = list(style)
        if len(styles) != frames.shape[0]:
            raise ValueError("style sequence must match the frame batch")
    if any(value not in ALL_STYLES for value in styles):
        raise ValueError("unknown text style")
    colour, brightness, horizontal, vertical, shapes = _descriptors(frames)
    sentences: list[str] = []
    for index, selected_style in enumerate(styles):
        c, b, h, v, shape = (
            colour[index], brightness[index], horizontal[index], vertical[index], shapes[index]
        )
        article = _article(c)
        if selected_style == 0:
            sentence = f"{article} {c} {shape} is visible on the {h} {v} side."
        elif selected_style == 1:
            sentence = f"the image shows {article} {b} {c} {shape}; it is {h} and {v}."
        elif selected_style == 2:
            sentence = f"look at the {c} {shape}, located {v} and {h} of centre."
        elif selected_style == 3:
            sentence = f"{h}, {v}: {article} {c} {shape} can be seen in the picture."
        else:
            sentence = f"there is {article} {c} {shape} in the picture, {v} {h}; it is {b}."
        encoded = sentence.encode("utf-8")
        if len(encoded) > text_length:
            raise ValueError("rendered grounded text exceeds text_length")
        sentences.append(sentence)
    result = torch.zeros(
        frames.shape[0], text_length, dtype=torch.long, device=frames.device
    )
    for index, sentence in enumerate(sentences):
        raw = torch.tensor(
            [value + 1 for value in sentence.encode("utf-8")],
            dtype=torch.long,
            device=frames.device,
        )
        result[index, : raw.numel()] = raw
    return result


class ByteTextFrontend(nn.Module):
    """A replaceable byte-to-event frontend with no semantic output fields."""

    def __init__(
        self,
        event_width: int,
        *,
        text_length: int = TEXT_LENGTH,
        embedding_width: int = 32,
        hidden_width: int = 96,
        position_bins: int = 0,
    ) -> None:
        super().__init__()
        if min(event_width, text_length, embedding_width, hidden_width) < 1:
            raise ValueError("frontend dimensions must be positive")
        if position_bins < 0:
            raise ValueError("position_bins cannot be negative")
        self.text_length = text_length
        self.embedding_width = embedding_width
        self.hidden_width = hidden_width
        self.position_bins = position_bins
        self.embedding = nn.Embedding(TEXT_VOCAB, embedding_width, padding_idx=0)
        # Character n-grams make recurring words recoverable at the short
        # qualification rung; max/mean pooling leaves word order available to
        # the raw text surface while avoiding a fixed word-ID vocabulary.
        convolution_width = max(16, hidden_width // 2)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                embedding_width, convolution_width, kernel_size=kernel,
                padding=kernel // 2,
            )
            for kernel in (3, 5, 7)
        )
        self.head = nn.Sequential(
            nn.Linear(convolution_width * 3 * 2 + embedding_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, event_width),
        )
        position_width = convolution_width * len(self.convolutions) * position_bins
        self.position_head = (
            nn.Sequential(
                nn.Linear(position_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, event_width),
            )
            if position_bins
            else None
        )
        if self.position_head is not None:
            # The new order path is a residual experiment. At initialization
            # it cannot damage the already-qualified n-gram frontend.
            nn.init.zeros_(self.position_head[-1].weight)
            nn.init.zeros_(self.position_head[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.text_length:
            raise ValueError("text tokens must have shape [batch, text_length]")
        if tokens.dtype != torch.long:
            raise ValueError("text tokens must be int64")
        if torch.any(tokens < 0) or torch.any(tokens >= TEXT_VOCAB):
            raise ValueError("text token is outside the byte vocabulary")
        embedded = self.embedding(tokens).transpose(1, 2)
        features = []
        position_features = []
        non_padding = tokens.ne(0).unsqueeze(1)
        relative_position = (
            tokens.ne(0).cumsum(dim=1).to(embedded.dtype) - 1
        ) / tokens.ne(0).sum(dim=1, keepdim=True).clamp_min(1).to(embedded.dtype)
        relative_position = relative_position.unsqueeze(1)
        for convolution in self.convolutions:
            values = F.gelu(convolution(embedded))
            pooled = values.masked_fill(~non_padding, -1e4).amax(dim=-1)
            mean = (values * non_padding).sum(dim=-1) / non_padding.sum(
                dim=-1
            ).clamp_min(1)
            features.extend((pooled, mean))
            if self.position_bins:
                for index in range(self.position_bins):
                    lower = index / self.position_bins
                    upper = (index + 1) / self.position_bins
                    position_mask = (
                        non_padding
                        & (relative_position >= lower)
                        & (relative_position < upper)
                    )
                    position_features.append(
                        (values * position_mask).sum(dim=-1)
                        / position_mask.sum(dim=-1).clamp_min(1)
                    )
        mean_embedding = self.embedding(tokens).masked_fill(
            tokens.eq(0).unsqueeze(-1), 0.0
        ).sum(dim=1) / tokens.ne(0).sum(dim=1, keepdim=True).clamp_min(1)
        state = torch.cat((*features, mean_embedding), dim=-1)
        output = self.head(state)
        if self.position_head is not None:
            output = output + self.position_head(torch.cat(position_features, dim=-1))
        return output


class ByteTransformerFrontend(nn.Module):
    """A small sequence-aware byte frontend for the representation frontier."""

    def __init__(
        self,
        event_width: int,
        *,
        text_length: int = TEXT_LENGTH,
        embedding_width: int = 64,
        attention_heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if min(event_width, text_length, embedding_width, attention_heads, layers) < 1:
            raise ValueError("transformer frontend dimensions must be positive")
        if embedding_width % attention_heads:
            raise ValueError("embedding_width must divide evenly across attention_heads")
        self.text_length = text_length
        self.embedding_width = embedding_width
        self.embedding = nn.Embedding(TEXT_VOCAB, embedding_width, padding_idx=0)
        self.position = nn.Embedding(text_length, embedding_width)
        block = nn.TransformerEncoderLayer(
            d_model=embedding_width,
            nhead=attention_heads,
            dim_feedforward=embedding_width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # MPS does not implement the nested-tensor mask probe used by the
        # optional fast path. Keep the ordinary padded attention path
        # portable instead of relying on an implicit CPU fallback.
        self.encoder = nn.TransformerEncoder(
            block,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_width),
            nn.Linear(embedding_width, embedding_width),
            nn.GELU(),
            nn.Linear(embedding_width, event_width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.text_length:
            raise ValueError("text tokens must have shape [batch, text_length]")
        if tokens.dtype != torch.long:
            raise ValueError("text tokens must be int64")
        if torch.any(tokens < 0) or torch.any(tokens >= TEXT_VOCAB):
            raise ValueError("text token is outside the byte vocabulary")
        positions = torch.arange(
            self.text_length, device=tokens.device, dtype=torch.long
        ).unsqueeze(0)
        padding = tokens.eq(0)
        encoded = self.encoder(
            self.embedding(tokens) + self.position(positions),
            src_key_padding_mask=padding,
        )
        non_padding = (~padding).unsqueeze(-1)
        pooled = (encoded * non_padding).sum(dim=1) / non_padding.sum(
            dim=1
        ).clamp_min(1)
        return self.head(pooled)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime(
    controller_path: Path,
    bus_path: Path,
    device: torch.device,
    *,
    position_bins: int = 0,
    frontend_kind: str = "cnn",
) -> tuple[AmodalControllerRuntime, nn.Module]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    if frontend_kind == "cnn":
        text: nn.Module = ByteTextFrontend(
            extracted.controller.width, position_bins=position_bins
        )
    elif frontend_kind == "transformer":
        if position_bins:
            raise ValueError("position_bins is only available for the CNN frontend")
        text = ByteTransformerFrontend(extracted.controller.width)
    else:
        raise ValueError(f"unknown frontend kind: {frontend_kind!r}")
    text = text.to(device)
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={"stream_a": copy.deepcopy(extracted.encoder), "stream_b": text},
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in text.parameters():
        parameter.requires_grad_(True)
    return runtime, text


@torch.no_grad()
def evaluate(
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str,
    style: int,
    text_renderer: TextRenderer = render_grounded_text,
) -> dict[str, float]:
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
    text = text_renderer(
        second.flatten(0, 1), style=style
    ).reshape(count, normal.trials, TEXT_LENGTH)
    contradictory_text = text_renderer(
        contradictory_second.flatten(0, 1), style=style,
    ).reshape(count, normal.trials, TEXT_LENGTH)

    def run(streams: tuple[tuple[torch.Tensor, str], ...]) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            feedback = torch.full_like(reward, float(trial == 1))
            named = {name: values[:, trial] for values, name in streams}
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    text_actions = run(((text, "stream_b"),))
    fused = run(((first, "stream_a"), (text, "stream_b")))
    shuffled = run(((first, "stream_a"), (text.roll(1, 0), "stream_b")))
    contradictory_actions = run(
        ((first, "stream_a"), (contradictory_text, "stream_b"))
    )
    full = run(((normal.frames, "stream_a"),))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float((actions[:, query] == normal.correct_actions[:, query]).float().mean())

    return {
        "text_only_accuracy": accuracy(text_actions),
        "fused_accuracy": accuracy(fused),
        "shuffled_partner_accuracy": accuracy(shuffled),
        "contradictory_partner_accuracy": accuracy(contradictory_actions),
        "contradictory_prediction_flip_rate": float(
            (fused[:, query] != contradictory_actions[:, query]).float().mean()
        ),
        "full_n1_accuracy": accuracy(full),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1001001)
    parser.add_argument("--updates", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--frontend",
        choices=("cnn", "transformer"),
        default="cnn",
        help="raw-byte frontend architecture for the representation frontier",
    )
    parser.add_argument(
        "--text-source",
        choices=(
            "pixel_template",
            "external_corpus",
            "external_corpus_v2",
            "external_annotation_table_v3",
        ),
        default="pixel_template",
        help="versioned caption source presented to the byte frontend",
    )
    parser.add_argument(
        "--position-bins",
        type=int,
        default=0,
        help="add content-relative order pooling bins to the byte frontend",
    )
    parser.add_argument(
        "--paired-text-views",
        action="store_true",
        help=(
            "align two different raw-text paraphrases of each same rendered "
            "scene to the frozen visual event"
        ),
    )
    parser.add_argument(
        "--diamond-replay",
        action="store_true",
        help=(
            "use a fixed balanced training cycle with one additional "
            "diamond appearance per update"
        ),
    )
    parser.add_argument("--device", default=("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 8 or args.eval_count < 64:
        raise ValueError("updates, batch-size, and eval-count are too small")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, text = load_runtime(
        args.controller,
        args.input_bus,
        device,
        position_bins=args.position_bins,
        frontend_kind=args.frontend,
    )
    text_renderer = {
        "pixel_template": render_grounded_text,
        "external_corpus": render_external_text,
        "external_corpus_v2": render_external_text_v2,
        "external_annotation_table_v3": render_external_annotation_text_v3,
    }[args.text_source]
    caption_corpus_digest = (
        corpus_sha256()
        if args.text_source == "external_corpus"
        else corpus_sha256(CORPUS_V2_PATH)
        if args.text_source == "external_corpus_v2"
        else corpus_sha256(ANNOTATION_TABLE_V3_PATH)
        if args.text_source == "external_annotation_table_v3"
        else None
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    optimizer = torch.optim.Adam(text.parameters(), lr=args.learning_rate)
    curve: list[dict[str, object]] = []
    paired_frames = 0
    training_appearances = (
        ("bars", "diamonds", "dot_pairs", "diamonds")
        if args.diamond_replay
        else ("bars", "diamonds", "dot_pairs")
    )
    training_started = perf_counter()
    for update in range(args.updates + 1):
        if update:
            runtime.train()
            losses = []
            for appearance_index, appearance in enumerate(training_appearances):
                batch = generate_lifetimes(
                    args.batch_size, 6,
                    seed=args.seed + update * 100 + appearance_index,
                    heldout=True, task="pair_relation", appearance=appearance,
                    support_trials=1, device=device,
                )
                _, second = split_complementary_views(batch.frames)
                frames = second.flatten(0, 1)
                style = TRAIN_STYLES[(update + appearance_index) % len(TRAIN_STYLES)]
                tokens = text_renderer(
                    frames, style=style
                )
                with torch.no_grad():
                    target = runtime.encoders["stream_a"](frames)
                losses.append(F.mse_loss(text(tokens), target))
                paired_frames += int(frames.shape[0])
                if args.paired_text_views:
                    paired_style = TRAIN_STYLES[
                        (TRAIN_STYLES.index(style) + 1) % len(TRAIN_STYLES)
                    ]
                    paired_tokens = text_renderer(
                        frames, style=paired_style
                    )
                    losses.append(F.mse_loss(text(paired_tokens), target))
                    paired_frames += int(frames.shape[0])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text.parameters(), 5.0)
            optimizer.step()
            runtime.eval()
        else:
            loss = None
        curve.append({
            "update": update,
            "paired_frames": paired_frames,
            "loss": None if loss is None else float(loss.detach()),
            "heldout_style": evaluate(
                runtime, count=args.eval_count, seed=args.seed,
                device=device, appearance="bars", style=HELDOUT_STYLES[0],
                text_renderer=text_renderer,
            ),
        })
    appearances = ("bars", "diamonds", "dot_pairs")
    final_by_appearance: dict[str, dict[str, dict[str, float]]] = {}
    for appearance_index, appearance in enumerate(appearances):
        final_by_appearance[appearance] = {}
        for style in ALL_STYLES:
            final_by_appearance[appearance][str(style)] = evaluate(
                runtime, count=args.eval_count * 2,
                seed=args.seed + 60_000 + appearance_index * 10_000 + style * 100,
                device=device, appearance=appearance, style=style,
                text_renderer=text_renderer,
            )
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    heldout_rows = [
        final_by_appearance[appearance][str(style)]
        for appearance in appearances for style in HELDOUT_STYLES
    ]
    passed = bool(
        controller_unchanged and all(
            row["fused_accuracy"] >= 0.90
            and row["shuffled_partner_accuracy"] <= 0.60
            and row["contradictory_partner_accuracy"] <= 0.25
            and row["contradictory_prediction_flip_rate"] >= 0.75
            and row["full_n1_accuracy"] >= 0.95
            for row in heldout_rows
        )
    )
    report = {
        "schema": "amodal-grounded-byte-text-alignment-v1",
        "claim": (
            "A byte-level grounded text frontend can align paraphrased visible "
            "descriptions into a frozen amodal neural-IR basis."
        ),
        "training_signal": "paired_encoded_event_consistency",
        "text_provenance": (
            "controlled_external_phrase_corpus_v2"
            if args.text_source == "external_corpus_v2"
            else "controlled_external_phrase_corpus"
            if args.text_source == "external_corpus"
            else "external_visible_pixel_description_source"
        ),
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
            "paired_text_views": args.paired_text_views,
            "diamond_replay": args.diamond_replay,
            "position_bins": args.position_bins,
            "frontend": args.frontend,
            "text_source": args.text_source,
            "caption_corpus_sha256": caption_corpus_digest,
            "training_appearances": training_appearances,
            "device": str(device),
            "text_length": TEXT_LENGTH,
            "byte_vocabulary": TEXT_VOCAB,
            "train_styles": TRAIN_STYLES,
            "heldout_styles": HELDOUT_STYLES,
            "trainable_components": ["byte_text_frontend"],
        },
        "accounting": {
            "unique_verifier_bits": 0,
            "unique_logical_lifetimes": 0,
            "optimizer_updates": args.updates,
            "paired_unlabeled_frames": paired_frames,
            "replayed_examples": 0,
            "wall_time_seconds": perf_counter() - training_started,
        },
        "curve": curve,
        "final_by_appearance": final_by_appearance,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "schema": "amodal-grounded-byte-text-frontend-v1",
            "controller_sha256": _sha256(args.controller),
            "input_bus_sha256": _sha256(args.input_bus),
            "latent_width": runtime.controller.width,
            "frontend_kind": (
                "byte_text_grounded_description_relative_order"
                if args.frontend == "cnn" and args.position_bins
                else f"byte_text_grounded_description_{args.frontend}"
            ),
            "position_bins": args.position_bins,
            "text_length": TEXT_LENGTH,
            "byte_vocabulary": TEXT_VOCAB,
            "train_styles": TRAIN_STYLES,
            "heldout_styles": HELDOUT_STYLES,
            "training_signal": report["training_signal"],
            "text_source": args.text_source,
            "caption_corpus_sha256": report["configuration"]["caption_corpus_sha256"],
            "source_report": str(args.report),
            "passed": passed,
            "state_dict": {name: value.detach().cpu().clone() for name, value in text.state_dict().items()},
        },
        args.adapter_out,
    )
    print(json.dumps({"passed": passed, "heldout": heldout_rows}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
