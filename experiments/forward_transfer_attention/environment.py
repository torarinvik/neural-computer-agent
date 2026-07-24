from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image, ImageDraw

from experiments.syllogimous_latent_agent.data import (
    IMAGE_HEIGHT, IMAGE_WIDTH, PCM_SAMPLES, PublicEpisode, render_public_audio,
)
from experiments.syllogimous_realtime.environment import Action, XorShift64


COLORS = ((239, 71, 111), (57, 189, 248), (72, 214, 136), (255, 196, 61))
SHOTS = (0, 1, 2, 4)


@dataclass(frozen=True)
class AttentionTransferLifetime:
    """Past colour mappings followed by a novel spatial-attention rule.

    Private metadata is used only by deterministic tests. The controller sees
    PublicEpisode RGB/PCM tensors, never mappings, feature IDs, rules, or labels.
    """

    studies: tuple[PublicEpisode, ...]
    old_queries: tuple[PublicEpisode, ...]
    old_audit_queries: tuple[PublicEpisode, ...]
    supports: tuple[PublicEpisode, ...]
    future_queries: tuple[PublicEpisode, ...]
    shots: tuple[int, ...]
    rule: int
    cue_code: int
    color_mapping: tuple[int, ...]
    support_features: tuple[tuple[int, int], ...]
    support_answers: tuple[int, ...]
    query_features: tuple[tuple[int, int], ...]
    seed: int


def _shuffle(values: list[int], rng: XorShift64) -> None:
    for index in range(len(values) - 1, 0, -1):
        other = rng.integer(0, index + 1)
        values[index], values[other] = values[other], values[index]


def _independent_choice(seed: int, heldout: bool, purpose: str, modulus: int) -> int:
    """Deterministic assignment without a low-complexity cross-field shortcut."""
    payload = f"attention-transfer-v1:{seed}:{int(heldout)}:{purpose}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def _shape(draw: ImageDraw.ImageDraw, shape: int, color: tuple[int, int, int],
           center: tuple[int, int], radius: int = 13) -> None:
    x, y = center
    box = (x - radius, y - radius, x + radius, y + radius)
    if shape % 2 == 0:
        draw.ellipse(box, fill=color, outline=(245, 245, 250), width=2)
    else:
        draw.rectangle(box, fill=color, outline=(245, 245, 250), width=2)


def _novel_cue(draw: ImageDraw.ImageDraw, code: int) -> None:
    # Meaning (attend left or right) is randomized each lifetime.
    x, y = 22, 20
    points = []
    for bit in range(6):
        angle = bit * np.pi / 3
        radius = 7 + 3 * ((code >> bit) & 1)
        points.append((int(round(x + np.cos(angle) * radius)),
                       int(round(y + np.sin(angle) * radius))))
    draw.polygon(points, fill=(215, 215, 235), outline=(255, 255, 255))
    if code & 64:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(25, 25, 35))


def _buttons(draw: ImageDraw.ImageDraw, answer: int | None) -> None:
    positions = tuple(int(round(18 + index * (IMAGE_WIDTH - 36) / 7)) for index in range(8))
    for index, x in enumerate(positions):
        draw.ellipse((x - 5, 77, x + 5, 87), outline=(100, 110, 135), width=1)
        if answer == index:
            draw.ellipse((x - 3, 79, x + 3, 85), fill=(255, 255, 255))


def _frame(seed: int, *, colors: tuple[int, ...], cue_code: int | None,
           answer: int | None, shapes: tuple[int, ...] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 25, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
                           outline=(75, 85, 110), width=2)
    if cue_code is not None:
        _novel_cue(draw, cue_code)
    centers = ((IMAGE_WIDTH // 2, 43),) if len(colors) == 1 else ((59, 43), (101, 43))
    for index, (color, center) in enumerate(zip(colors, centers)):
        # Shape varies independently and is therefore a genuine distractor.
        shape = int(rng.integers(0, 2)) if shapes is None else shapes[index]
        _shape(draw, shape, COLORS[color], center)
    _buttons(draw, answer)
    return np.asarray(image, dtype=np.uint8).copy()


def _mapping_frame(seed: int, mapping: tuple[int, ...],
                   color_ids: tuple[int, ...] | None = None,
                   line_width: int = 4) -> np.ndarray:
    """Publicly demonstrate both old associations on one non-symbolic card."""
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 25, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
                           outline=(75, 85, 110), width=2)
    if color_ids is None:
        color_ids = tuple(range(len(mapping)))
    if len(color_ids) != len(mapping):
        raise ValueError("one visible color identity per mapping is required")
    item_x = ((57, 103) if len(mapping) == 2 else
              tuple(int(round(28 + index * 104 / max(1, len(mapping) - 1)))
                    for index in range(len(mapping))))
    radius = 11 if len(mapping) == 2 else 9
    button_x = tuple(int(round(18 + index * (IMAGE_WIDTH - 36) / 7)) for index in range(8))
    for index, (color, action) in enumerate(zip(color_ids, mapping)):
        _shape(draw, color, COLORS[color], (item_x[index], 34), radius=radius)
        target_x = button_x[action]
        draw.line((item_x[index], 47, target_x, 74), fill=COLORS[color],
                  width=line_width)
    _buttons(draw, None)
    return np.asarray(image, dtype=np.uint8).copy()


def _episode(frame: np.ndarray, seed: int, action: int) -> PublicEpisode:
    unused = np.asarray([-1], dtype=np.int64)
    return PublicEpisode(frame[None], render_public_audio(seed, PCM_SAMPLES)[None],
                         np.asarray([action], dtype=np.int64), unused, unused.copy(),
                         unused.copy(), 1, seed, group=None)


def _sequence_episode(frames: list[np.ndarray], seed: int, action: int, *,
                      silent_audio: bool = False) -> PublicEpisode:
    length = len(frames)
    unused = np.full(length, -1, dtype=np.int64)
    return PublicEpisode(
        np.stack(frames),
        (np.zeros((length, PCM_SAMPLES), dtype=np.float32) if silent_audio else
         np.stack([render_public_audio(seed + index, PCM_SAMPLES)
                   for index in range(length)])),
        np.full(length, action, dtype=np.int64), unused, unused.copy(), unused.copy(),
        length, seed, group=None)


def generate_attention_lifetime(seed: int, *, heldout: bool = False,
                                query_count: int = 4) -> AttentionTransferLifetime:
    """Create a uniquely scored few-shot spatial-attention lifetime.

    Two colour-to-response associations change every lifetime. After learning
    them, a novel visual cue means "attend left" or "attend right". Support cards
    reveal the correct response visually. Future cards contain two differently
    coloured objects and require selecting the mapped response of the attended
    object. One support is sufficient in principle, but only if old knowledge is
    successfully retrieved and reused.
    """
    if query_count < 1 or query_count > 4:
        raise ValueError("query_count must be in 1..4")
    salt = 0xA0761D6478BD642F if heldout else 0
    rng = XorShift64(seed ^ salt)
    response_pool = list(range(8))
    _shuffle(response_pool, rng)
    color_mapping = response_pool[:2]
    rule = _independent_choice(seed, heldout, "rule", 2)
    # Task identity and rule are held out by seed/salt. Rendering style remains
    # matched in this first cognitive-atom benchmark; style transfer is audited
    # separately rather than confounded with learning-reuse.
    cue_code = _independent_choice(seed, heldout, "cue", 128)

    public_seed = seed * 10_000
    studies = [_episode(_mapping_frame(public_seed, tuple(color_mapping)),
                        public_seed, int(Action.NEXT))]

    old_queries = []
    old_audit_queries = []
    for color in range(2):
        public_seed = seed * 10_000 + 100 + color
        old_queries.append(_episode(
            _frame(public_seed, colors=(color,), cue_code=None, answer=None),
            public_seed, color_mapping[color]))
        audit_seed = seed * 10_000 + 150 + color
        old_audit_queries.append(_episode(
            _frame(audit_seed, colors=(color,), cue_code=None, answer=None),
            audit_seed, color_mapping[color]))

    pair_order = [(0, 1), (1, 0)]
    if rng.coin():
        pair_order.reverse()
    support_pairs = tuple(pair_order[index % 2] for index in range(4))
    query_pairs = tuple(pair_order[index % 2] for index in range(query_count))

    def answer_for(pair: tuple[int, int]) -> int:
        return color_mapping[pair[rule]]

    support_answers = tuple(answer_for(pair) for pair in support_pairs)
    supports = []
    for index, (pair, answer) in enumerate(zip(support_pairs, support_answers)):
        public_seed = seed * 10_000 + 200 + index
        supports.append(_episode(
            _frame(public_seed, colors=pair, cue_code=cue_code, answer=answer),
            public_seed, int(Action.NEXT)))

    future_queries = []
    for index, pair in enumerate(query_pairs):
        public_seed = seed * 10_000 + 300 + index
        future_queries.append(_episode(
            _frame(public_seed, colors=pair, cue_code=cue_code, answer=None),
            public_seed, answer_for(pair)))

    return AttentionTransferLifetime(
        tuple(studies), tuple(old_queries), tuple(old_audit_queries),
        tuple(supports), tuple(future_queries),
        SHOTS, rule, cue_code, tuple(color_mapping), support_pairs, support_answers,
        query_pairs, seed)


def generate_shape_attention_lifetime(seed: int, *, heldout: bool = False,
                                      query_count: int = 4) -> AttentionTransferLifetime:
    """Render a novel cue meaning attend circle or square, never left or right."""
    if query_count < 1 or query_count > 4:
        raise ValueError("query_count must be in 1..4")
    salt = 0xE7037ED1A0B428DB if heldout else 0
    rng = XorShift64(seed ^ salt)
    response_pool = list(range(8))
    _shuffle(response_pool, rng)
    color_mapping = response_pool[:2]
    target_shape = _independent_choice(seed, heldout, "shape-rule", 2)
    cue_code = _independent_choice(seed, heldout, "shape-cue", 128)
    base = seed * 10_000 + 5_000
    studies = [_episode(_mapping_frame(base, tuple(color_mapping)),
                        base, int(Action.NEXT))]
    old_queries, old_audit_queries = [], []
    for color in range(2):
        old_queries.append(_episode(
            _frame(base + 100 + color, colors=(color,), cue_code=None, answer=None),
            base + 100 + color, color_mapping[color]))
        old_audit_queries.append(_episode(
            _frame(base + 150 + color, colors=(color,), cue_code=None, answer=None),
            base + 150 + color, color_mapping[color]))
    color_orders = [(0, 1), (1, 0)]
    if rng.coin():
        color_orders.reverse()
    shape_orders = [(0, 1), (1, 0), (1, 0), (0, 1)]
    if rng.coin():
        shape_orders.reverse()
    support_colors = tuple(color_orders[index % 2] for index in range(4))
    query_colors = tuple(color_orders[(index + 1) % 2] for index in range(query_count))

    def answer_for(colors: tuple[int, int], shapes: tuple[int, int]) -> int:
        return color_mapping[colors[shapes.index(target_shape)]]

    support_answers, supports = [], []
    for index, (colors, shapes) in enumerate(zip(support_colors, shape_orders)):
        answer = answer_for(colors, shapes)
        support_answers.append(answer)
        public_seed = base + 200 + index
        supports.append(_episode(
            _frame(public_seed, colors=colors, shapes=shapes,
                   cue_code=cue_code, answer=answer),
            public_seed, int(Action.NEXT)))
    query_shapes = tuple(shape_orders[(index + 1) % 4] for index in range(query_count))
    future_queries = []
    for index, (colors, shapes) in enumerate(zip(query_colors, query_shapes)):
        public_seed = base + 300 + index
        future_queries.append(_episode(
            _frame(public_seed, colors=colors, shapes=shapes,
                   cue_code=cue_code, answer=None),
            public_seed, answer_for(colors, shapes)))
    return AttentionTransferLifetime(
        tuple(studies), tuple(old_queries), tuple(old_audit_queries),
        tuple(supports), tuple(future_queries), SHOTS, target_shape, cue_code,
        tuple(color_mapping), support_colors, tuple(support_answers), query_colors, seed)


def _temporal_object_frame(seed: int, color: int, cue_code: int) -> np.ndarray:
    """One instant in a temporal trial; order exists only across frames."""
    return _frame(seed, colors=(color,), cue_code=cue_code, answer=None,
                  shapes=(0,))


def _temporal_feedback_frame(seed: int, cue_code: int, answer: int, *,
                             rewarded_color: int | None = None,
                             feedback_mode: str = "white-button") -> np.ndarray:
    """Visual response feedback without replaying either ordered object."""
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 25, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
                           outline=(75, 85, 110), width=2)
    _novel_cue(draw, cue_code)
    if feedback_mode == "white-button":
        _buttons(draw, answer)
    elif feedback_mode == "color-button":
        if rewarded_color is None:
            raise ValueError("color-button feedback requires rewarded_color")
        _buttons(draw, None)
        positions = tuple(int(round(18 + index * (IMAGE_WIDTH - 36) / 7))
                          for index in range(8))
        x = positions[answer]
        draw.ellipse((x - 4, 78, x + 4, 86), fill=COLORS[rewarded_color],
                     outline=(255, 255, 255), width=1)
    elif feedback_mode == "color-object":
        if rewarded_color is None:
            raise ValueError("color-object feedback requires rewarded_color")
        return _frame(
            seed, colors=(rewarded_color,), cue_code=cue_code,
            answer=None, shapes=(0,))
    else:
        raise ValueError(f"unknown temporal feedback mode {feedback_mode!r}")
    return np.asarray(image, dtype=np.uint8).copy()


def generate_compositional_temporal_attention_lifetime(
        seed: int, *, heldout: bool = False,
        query_count: int = 4,
        feedback_mode: str = "white-button") -> AttentionTransferLifetime:
    """Hard temporal level: learn four mappings and a first/last rule together."""
    if query_count < 1 or query_count > 4:
        raise ValueError("query_count must be in 1..4")
    salt = 0x8EBC6AF09C88C6E3 if heldout else 0
    rng = XorShift64(seed ^ salt)
    response_pool = list(range(8))
    _shuffle(response_pool, rng)
    color_mapping = response_pool[:4]
    target_time = _independent_choice(seed, heldout, "temporal-rule", 2)
    cue_code = _independent_choice(seed, heldout, "temporal-cue", 128)
    base = seed * 10_000 + 7_500
    studies = [
        _episode(_mapping_frame(base, tuple(color_mapping[:2]), (0, 1)),
                 base, int(Action.NEXT)),
        _episode(_mapping_frame(base + 1, tuple(color_mapping[2:]), (2, 3)),
                 base + 1, int(Action.NEXT)),
    ]
    old_queries, old_audit_queries = [], []
    for color in range(4):
        old_queries.append(_episode(
            _frame(base + 100 + color, colors=(color,), cue_code=None, answer=None),
            base + 100 + color, color_mapping[color]))
        old_audit_queries.append(_episode(
            _frame(base + 150 + color, colors=(color,), cue_code=None, answer=None),
            base + 150 + color, color_mapping[color]))
    colors = list(range(4))
    _shuffle(colors, rng)
    a, b, c, d = colors
    support_orders = ((a, b), (c, d), (b, d), (a, c))
    # No query repeats the first demonstrated pair in either order. A selected
    # color from support therefore cannot be reused as a shortcut.
    query_bank = ((c, a), (d, b), (d, a), (b, c))
    query_orders = tuple(query_bank[index] for index in range(query_count))

    def answer_for(order: tuple[int, int]) -> int:
        return color_mapping[order[target_time]]

    support_answers, supports = [], []
    for index, order in enumerate(support_orders):
        answer = answer_for(order)
        support_answers.append(answer)
        public_seed = base + 200 + index * 10
        frames = [
            _temporal_object_frame(base, order[0], cue_code),
            _temporal_object_frame(base, order[1], cue_code),
            _temporal_feedback_frame(
                public_seed + 2, cue_code, answer,
                rewarded_color=order[target_time], feedback_mode=feedback_mode),
        ]
        supports.append(_sequence_episode(
            frames, public_seed, int(Action.NEXT), silent_audio=True))
    future_queries = []
    for index, order in enumerate(query_orders):
        public_seed = base + 300 + index * 10
        frames = [
            _temporal_object_frame(base, order[0], cue_code),
            _temporal_object_frame(base, order[1], cue_code),
        ]
        future_queries.append(_sequence_episode(
            frames, public_seed, answer_for(order), silent_audio=True))
    return AttentionTransferLifetime(
        tuple(studies), tuple(old_queries), tuple(old_audit_queries), tuple(supports),
        tuple(future_queries), SHOTS, target_time, cue_code, tuple(color_mapping),
        support_orders, tuple(support_answers), query_orders, seed)


def generate_temporal_attention_lifetime(seed: int, *, heldout: bool = False,
                                         query_count: int = 4,
                                         grounded_rule: bool = False,
                                         forced_rule: int | None = None,
                                         mapping_line_width: int = 4,
                                         feedback_mode: str = "white-button",
                                         render_seed: int | None = None,
                                         color_ids: tuple[int, int] = (0, 1),
                                         ) -> AttentionTransferLifetime:
    """Curriculum atom that isolates learning a first-versus-last rule."""
    if query_count < 1 or query_count > 4:
        raise ValueError("query_count must be in 1..4")
    salt = 0x8EBC6AF09C88C6E3 if heldout else 0
    rng = XorShift64(seed ^ salt)
    response_pool = list(range(8))
    _shuffle(response_pool, rng)
    color_mapping = response_pool[:2]
    if forced_rule not in (None, 0, 1):
        raise ValueError("forced_rule must be None, 0, or 1")
    if len(color_ids) != 2 or len(set(color_ids)) != 2:
        raise ValueError("color_ids must contain two distinct colors")
    target_time = (_independent_choice(seed, heldout, "temporal-atom-rule", 2)
                   if forced_rule is None else forced_rule)
    cue_code = ((19, 108)[target_time] if grounded_rule else
                _independent_choice(seed, heldout, "temporal-atom-cue", 128))
    # Optional nuisance-only augmentation: task metadata remains a function of
    # seed, while backgrounds/audio/public episode seeds come from render_seed.
    base = (seed if render_seed is None else render_seed) * 10_000 + 7_500
    studies = [_episode(_mapping_frame(
                            base, tuple(color_mapping), tuple(color_ids),
                            line_width=mapping_line_width),
                        base, int(Action.NEXT))]
    old_queries, old_audit_queries = [], []
    for color in range(2):
        old_queries.append(_episode(
            _frame(base + 100 + color, colors=(color_ids[color],),
                   cue_code=None, answer=None),
            base + 100 + color, color_mapping[color]))
        old_audit_queries.append(_episode(
            _frame(base + 150 + color, colors=(color_ids[color],),
                   cue_code=None, answer=None),
            base + 150 + color, color_mapping[color]))
    orientations = [(0, 1), (1, 0)]
    if rng.coin():
        orientations.reverse()
    support_orders = tuple(orientations[index % 2] for index in range(4))
    # One-shot queries begin with the opposite orientation. A constant response
    # copied from the demonstration is therefore guaranteed to be wrong.
    query_orders = tuple(orientations[(index + 1) % 2]
                         for index in range(query_count))

    def answer_for(order: tuple[int, int]) -> int:
        return color_mapping[order[target_time]]

    support_answers, supports = [], []
    for index, order in enumerate(support_orders):
        answer = answer_for(order)
        support_answers.append(answer)
        public_seed = base + 200 + index * 10
        supports.append(_sequence_episode([
            _temporal_object_frame(base, color_ids[order[0]], cue_code),
            _temporal_object_frame(base, color_ids[order[1]], cue_code),
            _temporal_feedback_frame(
                public_seed + 2, cue_code, answer,
                rewarded_color=color_ids[order[target_time]],
                feedback_mode=feedback_mode),
        ], public_seed, int(Action.NEXT), silent_audio=True))
    future_queries = []
    for index, order in enumerate(query_orders):
        public_seed = base + 300 + index * 10
        future_queries.append(_sequence_episode([
            _temporal_object_frame(base, color_ids[order[0]], cue_code),
            _temporal_object_frame(base, color_ids[order[1]], cue_code),
        ], public_seed, answer_for(order), silent_audio=True))
    return AttentionTransferLifetime(
        tuple(studies), tuple(old_queries), tuple(old_audit_queries), tuple(supports),
        tuple(future_queries), SHOTS, target_time, cue_code, tuple(color_mapping),
        support_orders, tuple(support_answers), query_orders, seed)


def generate_temporal_grounding_lifetime(seed: int, *, heldout: bool = False,
                                         query_count: int = 4
                                         ) -> AttentionTransferLifetime:
    """Pretraining level with stable visual cues for first and last selection."""
    return generate_temporal_attention_lifetime(
        seed, heldout=heldout, query_count=query_count, grounded_rule=True)


def generate_temporal_first_lifetime(seed: int, *, heldout: bool = False,
                                     query_count: int = 4
                                     ) -> AttentionTransferLifetime:
    return generate_temporal_attention_lifetime(
        seed, heldout=heldout, query_count=query_count,
        grounded_rule=True, forced_rule=0)


def generate_temporal_last_lifetime(seed: int, *, heldout: bool = False,
                                    query_count: int = 4
                                    ) -> AttentionTransferLifetime:
    return generate_temporal_attention_lifetime(
        seed, heldout=heldout, query_count=query_count,
        grounded_rule=True, forced_rule=1)
