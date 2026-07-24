"""Render what the temporal pixel probe sees before and after 4x downsampling."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .environment import generate_temporal_attention_lifetime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17_000_000)
    args = parser.parse_args()
    item = generate_temporal_attention_lifetime(args.seed)
    frames = np.concatenate((item.studies[0].frames, item.supports[0].frames), axis=0)
    labels = ("mapping card", "first object", "second object", "answer feedback")
    canvas = Image.new("RGB", (160 * 4, 96 * 2 + 34), (12, 12, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(frames, labels)):
        raw = Image.fromarray(array)
        reduced = raw.resize((40, 24), Image.Resampling.BILINEAR)
        restored = reduced.resize((160, 96), Image.Resampling.NEAREST)
        canvas.paste(raw, (index * 160, 18))
        canvas.paste(restored, (index * 160, 114))
        draw.text((index * 160 + 4, 3), label, fill=(235, 235, 245))
    draw.text((4, 99), "raw 160x96", fill=(235, 235, 245))
    draw.text((4, 195), "bilinear 40x24, enlarged", fill=(235, 235, 245))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
