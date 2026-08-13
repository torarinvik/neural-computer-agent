"""Two-seed audit of a generic recipe-basis extension.

The learner is trained only on randomly generated programs over abstract
register slots.  No benchmark task family, semantic label, or privileged
answer is present in its weights.  The audit separates three questions:

* can the fixed interpreter execute unseen programs from the old basis?
* does a generic parallel-composition extension preserve that ability?
* can the interpreter preserve an explicit modulus across mixed value
  domains?
* can a parallel composition be learned when the exact target is held out?

This is deliberately a small proving ground for the controller-as-CPU
boundary.  It does not claim that a neural interpreter is already the
canonical controller or that the basis is sufficient for general cognition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from neural_computer.recipe_basis import RecipeBasis, RecipeInstruction

SLOTS = 6
VALUES = 8
SLOT_VALUES = (2, 2, 8, 8, 8, 8)
ATOMIC_OPS = ("noop", "inc", "dec", "cinc", "cdec", "copy", "swap")
OP_TO_ID = {name: index for index, name in enumerate(ATOMIC_OPS)}
MODULUS_FEATURE_OFFSET = len(ATOMIC_OPS) + SLOTS + SLOTS
ATOMIC_FEATURE_WIDTH = MODULUS_FEATURE_OFFSET + VALUES
INSTRUCTION_FEATURE_WIDTH = ATOMIC_FEATURE_WIDTH * 2


def _randint(generator: torch.Generator, high: int) -> int:
    return int(torch.randint(high, (), generator=generator).item())


def _sample_slot_values(
    generator: torch.Generator,
    slot_values: tuple[int, ...],
    *,
    targets: dict[int, int] | None = None,
) -> tuple[int, ...]:
    """Permute domains, optionally pinning domains for probe target slots."""

    targets = {} if targets is None else dict(targets)
    if any(slot < 0 or slot >= len(slot_values) for slot in targets):
        raise ValueError("domain target slot is outside the register")
    remaining = list(slot_values)
    for modulus in targets.values():
        try:
            remaining.remove(modulus)
        except ValueError as error:
            raise ValueError("target domain is absent from the domain profile") from error
    order = torch.randperm(len(remaining), generator=generator).tolist()
    shuffled = [remaining[index] for index in order]
    result: list[int | None] = [None] * len(slot_values)
    for slot, modulus in targets.items():
        result[slot] = modulus
    for slot, modulus in enumerate(result):
        if modulus is None:
            result[slot] = shuffled.pop()
    return tuple(int(modulus) for modulus in result)


def _sample_initial(
    generator: torch.Generator,
    *,
    batch_size: int,
    slot_values: tuple[int, ...],
) -> torch.Tensor:
    return torch.tensor(
        [
            [_randint(generator, modulus) for modulus in slot_values]
            for _ in range(batch_size)
        ],
        dtype=torch.long,
    )


def modulus_boundary(
    *,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    legacy_modulus: int = VALUES,
) -> dict[str, object]:
    """Measure the legacy global-modulus mismatch without training a model."""

    legacy_rates: list[float] = []
    explicit_rates: list[float] = []
    for modulus in slot_values:
        states = tuple(range(modulus))
        expected = tuple((value + 1) % modulus for value in states)
        legacy = tuple((value + 1) % legacy_modulus for value in states)
        explicit = tuple((value + 1) % modulus for value in states)
        legacy_rates.append(
            sum(actual == want for actual, want in zip(legacy, expected, strict=True))
            / len(states)
        )
        explicit_rates.append(
            sum(actual == want for actual, want in zip(explicit, expected, strict=True))
            / len(states)
        )
    return {
        "slot_values": slot_values,
        "legacy_global_modulus": legacy_modulus,
        "legacy_match_rates": legacy_rates,
        "explicit_match_rates": explicit_rates,
    }


def _atomic_instruction(
    generator: torch.Generator,
    *,
    slot_values: tuple[int, ...] = SLOT_VALUES,
) -> RecipeInstruction:
    op = ATOMIC_OPS[_randint(generator, len(ATOMIC_OPS))]
    if op == "noop":
        return RecipeInstruction(op)
    first = _randint(generator, SLOTS)
    if op in {"cinc", "cdec", "copy", "swap"}:
        compatible = tuple(
            second
            for second in range(SLOTS)
            if second != first
            and (
                op in {"cinc", "cdec"}
                or (
                    op == "copy"
                    and slot_values[second] <= slot_values[first]
                )
                or (op == "swap" and slot_values[second] == slot_values[first])
            )
        )
        if not compatible:
            return RecipeInstruction("inc", first, modulus=slot_values[first])
        second = compatible[_randint(generator, len(compatible))]
        if op in {"cinc", "cdec"}:
            return RecipeInstruction(
                op,
                first,
                second,
                modulus=slot_values[first],
            )
        return RecipeInstruction(op, first, second)
    return RecipeInstruction(op, first, modulus=slot_values[first])


def _instruction(
    generator: torch.Generator,
    *,
    allow_parallel: bool,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    holdout_parallel_target: bool = False,
) -> RecipeInstruction:
    if allow_parallel and _randint(generator, 5) == 0:
        for _ in range(64):
            left = _atomic_instruction(generator, slot_values=slot_values)
            right = _atomic_instruction(generator, slot_values=slot_values)
            candidate = RecipeInstruction("parallel", children=(left, right))
            if holdout_parallel_target and _is_parallel_target(candidate):
                continue
            if (
                left.written_slots()
                and right.written_slots()
                and not left.written_slots() & right.written_slots()
            ):
                return candidate
        # A deterministic fallback keeps the sampler total if the random
        # attempts happen to choose overlapping effects.
        return RecipeInstruction(
            "parallel",
            children=(
                RecipeInstruction("inc", 0, modulus=slot_values[0]),
                RecipeInstruction("inc", 2, modulus=slot_values[2]),
            ),
        )
    return _atomic_instruction(generator, slot_values=slot_values)


def _is_parallel_target(instruction: RecipeInstruction) -> bool:
    if instruction.op != "parallel" or instruction.children is None:
        return False
    first, second = instruction.children
    if first.op != "inc" or second.op != "inc":
        return False
    if first.modulus != 2 or second.modulus != 2:
        return False
    return {first.first, second.first} == {0, 1}


def _atomic_features(instruction: RecipeInstruction) -> torch.Tensor:
    if instruction.op not in OP_TO_ID:
        raise ValueError("parallel children must be atomic")
    feature = torch.zeros(ATOMIC_FEATURE_WIDTH, dtype=torch.float32)
    feature[OP_TO_ID[instruction.op]] = 1.0
    if instruction.first is not None:
        feature[len(ATOMIC_OPS) + instruction.first] = 1.0
    if instruction.second is not None:
        feature[len(ATOMIC_OPS) + SLOTS + instruction.second] = 1.0
    if instruction.modulus is not None:
        feature[MODULUS_FEATURE_OFFSET + instruction.modulus - 2] = 1.0
    return feature


def instruction_features(instruction: RecipeInstruction) -> torch.Tensor:
    """Encode an instruction without assigning meaning to latent coordinates."""

    if instruction.op == "parallel":
        assert instruction.children is not None
        return torch.cat(tuple(_atomic_features(child) for child in instruction.children))
    return torch.cat((_atomic_features(instruction), torch.zeros(ATOMIC_FEATURE_WIDTH)))


@dataclass(frozen=True)
class Batch:
    initial: torch.Tensor
    instructions: torch.Tensor
    targets: torch.Tensor


def sample_batch(
    *,
    generator: torch.Generator,
    batch_size: int,
    length: int,
    allow_parallel: bool,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    holdout_parallel_target: bool = False,
) -> Batch:
    domain_rows = tuple(
        _sample_slot_values(generator, slot_values)
        if randomize_domains
        else slot_values
        for _ in range(batch_size)
    )
    initial_rows = tuple(
        _sample_initial(
            generator,
            batch_size=1,
            slot_values=row_values,
        )[0]
        for row_values in domain_rows
    )
    initial = torch.stack(initial_rows, dim=0)
    feature_rows: list[list[torch.Tensor]] = []
    target_rows: list[list[tuple[int, ...]]] = []
    for row, row_slot_values in zip(initial.tolist(), domain_rows, strict=True):
        state = tuple(int(value) for value in row)
        row_features: list[torch.Tensor] = []
        row_targets: list[tuple[int, ...]] = []
        for _ in range(length):
            instruction = _instruction(
                generator,
                allow_parallel=allow_parallel,
                slot_values=row_slot_values,
                holdout_parallel_target=holdout_parallel_target,
            )
            row_features.append(instruction_features(instruction))
            state = instruction.apply(state, values=row_slot_values)
            row_targets.append(state)
        feature_rows.append(row_features)
        target_rows.append(row_targets)
    return Batch(
        initial=initial,
        instructions=torch.stack(
            [torch.stack(row) for row in feature_rows], dim=0
        ),
        targets=torch.tensor(target_rows, dtype=torch.long),
    )


class LearnedRecipeInterpreter(nn.Module):
    """A fixed recurrent interpreter for opaque instruction vectors."""

    def __init__(self, *, hidden: int = 128) -> None:
        super().__init__()
        self.value_encoder = nn.Linear(VALUES, hidden)
        self.position = nn.Embedding(SLOTS, hidden)
        self.instruction_encoder = nn.Linear(INSTRUCTION_FEATURE_WIDTH, hidden)
        self.interpreter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ),
            num_layers=2,
        )
        self.output = nn.Linear(hidden, VALUES)

    def forward(
        self,
        initial: torch.Tensor,
        instructions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        state = F.one_hot(initial, VALUES).to(torch.float32)
        outputs: list[torch.Tensor] = []
        positions = self.position(torch.arange(SLOTS, device=state.device)).unsqueeze(0)
        for step in range(instructions.shape[1]):
            state_tokens = self.value_encoder(state) + positions
            instruction_token = self.instruction_encoder(instructions[:, step]).unsqueeze(1)
            tokens = self.interpreter(torch.cat((instruction_token, state_tokens), dim=1))
            logits = self.output(tokens[:, 1:])
            outputs.append(logits)
            state = logits.softmax(dim=-1)
        return tuple(outputs)


def _loss_and_accuracy(
    model: LearnedRecipeInterpreter,
    batch: Batch,
) -> tuple[torch.Tensor, float]:
    outputs = model(batch.initial, batch.instructions)
    losses = tuple(
        F.cross_entropy(output.reshape(-1, VALUES), target.reshape(-1))
        for output, target in zip(outputs, batch.targets.transpose(0, 1), strict=True)
    )
    prediction = outputs[-1].argmax(dim=-1)
    exact = prediction.eq(batch.targets[:, -1]).all(dim=-1).float().mean()
    return torch.stack(losses).mean(), float(exact.item())


@torch.no_grad()
def evaluate(
    model: LearnedRecipeInterpreter,
    *,
    seed: int,
    allow_parallel: bool,
    length: int,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    holdout_parallel_target: bool = False,
    batches: int = 8,
    batch_size: int = 128,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(batches):
        batch = sample_batch(
            generator=generator,
            batch_size=batch_size,
            length=length,
            allow_parallel=allow_parallel,
            slot_values=slot_values,
            randomize_domains=randomize_domains,
            holdout_parallel_target=holdout_parallel_target,
        )
        scores.append(_loss_and_accuracy(model, batch)[1])
    return float(sum(scores) / len(scores))


@torch.no_grad()
def evaluate_parallel_target(
    model: LearnedRecipeInterpreter,
    *,
    seed: int,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    holdout_parallel_target: bool = False,
    batch_size: int = 128,
    batches: int = 8,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    instruction = RecipeInstruction(
        "parallel",
        children=(
            RecipeInstruction("inc", 0, modulus=slot_values[0]),
            RecipeInstruction("inc", 1, modulus=slot_values[1]),
        ),
    )
    features = instruction_features(instruction).view(1, 1, -1)
    scores: list[float] = []
    for _ in range(batches):
        row_slot_values = _sample_slot_values(
            generator,
            slot_values,
            targets={0: slot_values[0], 1: slot_values[1]}
            if randomize_domains
            else None,
        )
        initial = _sample_initial(
            generator,
            batch_size=batch_size,
            slot_values=row_slot_values,
        )
        targets = initial.clone()
        targets[:, 0] = (targets[:, 0] + 1) % row_slot_values[0]
        targets[:, 1] = (targets[:, 1] + 1) % row_slot_values[1]
        output = model(initial, features.expand(batch_size, -1, -1))[-1]
        scores.append(float(output.argmax(-1).eq(targets).all(-1).float().mean()))
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_arithmetic_target(
    model: LearnedRecipeInterpreter,
    *,
    seed: int,
    operation: str,
    target_slot: int,
    instruction_modulus: int | None = None,
    condition_slot: int | None = None,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    holdout_parallel_target: bool = False,
    batch_size: int = 128,
    batches: int = 8,
) -> float:
    """Evaluate one opaque arithmetic target without parallel composition."""

    if operation not in {"inc", "dec", "cinc", "cdec"}:
        raise ValueError("unsupported arithmetic probe operation")
    if operation in {"cinc", "cdec"}:
        if condition_slot is None or condition_slot == target_slot:
            raise ValueError("conditional arithmetic needs a distinct condition slot")
    elif condition_slot is not None:
        raise ValueError("unconditional arithmetic cannot have a condition slot")
    generator = torch.Generator().manual_seed(seed)
    modulus = slot_values[target_slot]
    instruction = RecipeInstruction(
        operation,  # type: ignore[arg-type]
        target_slot,
        condition_slot,
        modulus=modulus if instruction_modulus is None else instruction_modulus,
    )
    features = instruction_features(instruction).view(1, 1, -1)
    scores: list[float] = []
    for _ in range(batches):
        row_slot_values = _sample_slot_values(
            generator,
            slot_values,
            targets={target_slot: modulus} if randomize_domains else None,
        )
        initial = _sample_initial(
            generator,
            batch_size=batch_size,
            slot_values=row_slot_values,
        )
        targets = initial.clone()
        if operation in {"inc", "cinc"}:
            delta = 1
        else:
            delta = -1
        if operation in {"cinc", "cdec"}:
            assert condition_slot is not None
            active = initial[:, condition_slot] != 0
        else:
            active = torch.ones(batch_size, dtype=torch.bool)
        updated = targets[:, target_slot].clone()
        updated[active] = (
            updated[active] + delta
        ) % row_slot_values[target_slot]
        targets[:, target_slot] = updated
        output = model(initial, features.expand(batch_size, -1, -1))[-1]
        scores.append(float(output.argmax(-1).eq(targets).all(-1).float().mean()))
    return sum(scores) / len(scores)


def evaluate_single_modulus_target(
    model: LearnedRecipeInterpreter,
    *,
    seed: int,
    target_slot: int,
    instruction_modulus: int | None = None,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    batch_size: int = 128,
    batches: int = 8,
) -> float:
    """Evaluate one opaque increment target without parallel composition."""

    return evaluate_arithmetic_target(
        model,
        seed=seed,
        operation="inc",
        target_slot=target_slot,
        instruction_modulus=instruction_modulus,
        slot_values=slot_values,
        randomize_domains=randomize_domains,
        batch_size=batch_size,
        batches=batches,
    )


def train_arm(
    *,
    seed: int,
    allow_parallel: bool,
    slot_values: tuple[int, ...] = SLOT_VALUES,
    randomize_domains: bool = True,
    holdout_parallel_target: bool = False,
    updates: int,
    batch_size: int,
    hidden: int,
    eval_every: int,
) -> dict[str, object]:
    started = perf_counter()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = LearnedRecipeInterpreter(hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 10_000)
    curve: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        batch = sample_batch(
            generator=generator,
            batch_size=batch_size,
            length=2,
            allow_parallel=allow_parallel,
            slot_values=slot_values,
            randomize_domains=randomize_domains,
            holdout_parallel_target=holdout_parallel_target,
        )
        loss, _ = _loss_and_accuracy(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            curve.append(
                {
                    "update": update,
                    "base_unseen_length_2": evaluate(
                        model,
                        seed=seed + 20_000 + update,
                        allow_parallel=False,
                        length=2,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "base_unseen_double_length_4": evaluate(
                        model,
                        seed=seed + 30_000 + update,
                        allow_parallel=False,
                        length=4,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "parallel_target": evaluate_parallel_target(
                        model,
                        seed=seed + 40_000 + update,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "modulus_target_m2": evaluate_single_modulus_target(
                        model,
                        seed=seed + 50_000 + update,
                        target_slot=0,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "modulus_target_m8": evaluate_single_modulus_target(
                        model,
                        seed=seed + 60_000 + update,
                        target_slot=2,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "modulus_target_m2_wrong_m8": evaluate_single_modulus_target(
                        model,
                        seed=seed + 70_000 + update,
                        target_slot=0,
                        instruction_modulus=VALUES,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                    ),
                    "dec_target_m2": evaluate_arithmetic_target(
                        model,
                        seed=seed + 80_000 + update,
                        operation="dec",
                        target_slot=0,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                    "dec_target_m8": evaluate_arithmetic_target(
                        model,
                        seed=seed + 90_000 + update,
                        operation="dec",
                        target_slot=2,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                    "cinc_target_m2": evaluate_arithmetic_target(
                        model,
                        seed=seed + 100_000 + update,
                        operation="cinc",
                        target_slot=0,
                        condition_slot=1,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                    "cinc_target_m8": evaluate_arithmetic_target(
                        model,
                        seed=seed + 110_000 + update,
                        operation="cinc",
                        target_slot=2,
                        condition_slot=3,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                    "cdec_target_m2": evaluate_arithmetic_target(
                        model,
                        seed=seed + 120_000 + update,
                        operation="cdec",
                        target_slot=0,
                        condition_slot=1,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                    "cdec_target_m8": evaluate_arithmetic_target(
                        model,
                        seed=seed + 130_000 + update,
                        operation="cdec",
                        target_slot=2,
                        condition_slot=3,
                        slot_values=slot_values,
                        randomize_domains=randomize_domains,
                        batches=4,
                    ),
                }
            )
    def stable_update(key: str, threshold: float = 0.9) -> int | None:
        for index, row in enumerate(curve):
            if all(
                float(later[key]) >= threshold for later in curve[index:]
            ):
                return int(row["update"])
        return None

    return {
        "seed": seed,
        "allow_parallel": allow_parallel,
        "holdout_parallel_target": holdout_parallel_target,
        "updates": updates,
        "batch_size": batch_size,
        "hidden": hidden,
        "curve": curve,
        "stable_updates_at_0_9": {
            key: stable_update(key)
            for key in (
                "base_unseen_length_2",
                "base_unseen_double_length_4",
                "parallel_target",
                "modulus_target_m2",
                "modulus_target_m8",
                "dec_target_m2",
                "dec_target_m8",
                "cinc_target_m2",
                "cinc_target_m8",
                "cdec_target_m2",
                "cdec_target_m8",
            )
        },
        "accounting": {
            "unique_program_steps": updates * batch_size * 2,
            "optimizer_updates": updates,
            "replayed_examples": 0,
            "wall_time_seconds": perf_counter() - started,
        },
        "final": curve[-1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    arms = tuple(
        arm
        for seed in (args.seed, args.seed + 1)
        for arm in (
            train_arm(
                seed=seed,
                allow_parallel=allow_parallel,
                slot_values=SLOT_VALUES,
                randomize_domains=True,
                holdout_parallel_target=allow_parallel,
                updates=args.updates,
                batch_size=args.batch_size,
                hidden=args.hidden,
                eval_every=args.eval_every,
            )
            for allow_parallel in (False, True)
        )
    )
    return {
        "schema": "neural-computer.recipe-expressibility-audit.v4",
        "source": "in_repository_run",
        "configuration": {
            "slots": SLOTS,
            "values": VALUES,
            "slot_values": SLOT_VALUES,
            "randomized_slot_domains": True,
            "parallel_target_held_out_from_parallel_training": True,
            "training_program_length": 2,
            "double_length": 4,
            "random_programs_only": True,
            "controller_task_labels": False,
        },
        "modulus_boundary": modulus_boundary(),
        "arms": arms,
        "symbolic_boundary": {
            "baseline": RecipeBasis(
                slot_values=SLOT_VALUES,
                allow_parallel=False,
            )
            .expressibility_probe(
                lambda state: tuple(
                    (value + 1) % 2 if index in (0, 1) else value
                    for index, value in enumerate(state)
                ),
                states=tuple(
                    (first, second, 0, 1, 0, 1)
                    for first in range(SLOT_VALUES[0])
                    for second in range(SLOT_VALUES[1])
                ),
            )
            .status,
            "parallel": RecipeBasis(
                slot_values=SLOT_VALUES,
                allow_parallel=True,
            )
            .expressibility_probe(
                lambda state: tuple(
                    (value + 1) % 2 if index in (0, 1) else value
                    for index, value in enumerate(state)
                ),
                states=tuple(
                    (first, second, 0, 1, 0, 1)
                    for first in range(SLOT_VALUES[0])
                    for second in range(SLOT_VALUES[1])
                ),
            )
            .status,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70421)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = run(args)
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.json is not None:
        args.json.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
