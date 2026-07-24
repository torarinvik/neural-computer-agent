"""Jointly adapt perception/controller and consolidation to a new primitive."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
import time
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent, parameter_count

from .consolidator import LatentConsolidator
from .environment import (generate_attention_lifetime,
                          generate_shape_attention_lifetime,
                          generate_temporal_first_lifetime,
                          generate_temporal_attention_lifetime,
                          generate_temporal_grounding_lifetime,
                          generate_temporal_last_lifetime)
from .train import seed_everything
from .train_consolidator import evaluate, run_compaction_batch


GENERATORS = {
    "spatial": generate_attention_lifetime,
    "shape": generate_shape_attention_lifetime,
    "temporal": generate_temporal_attention_lifetime,
}


def _add_temporal_counterfactual_queries(lifetime):
    """Add verifier-labeled reversed queries without exposing metadata to the model."""
    reversed_orders = []
    reversed_queries = []
    for order, episode in zip(lifetime.query_features, lifetime.future_queries):
        reversed_order = (order[1], order[0])
        answer = lifetime.color_mapping[reversed_order[lifetime.rule]]
        reversed_orders.append(reversed_order)
        reversed_queries.append(replace(
            episode,
            frames=episode.frames[::-1].copy(),
            pcm=episode.pcm[::-1].copy(),
            actions=episode.actions * 0 + answer))
    return replace(
        lifetime,
        future_queries=tuple(lifetime.future_queries) + tuple(reversed_queries),
        query_features=tuple(lifetime.query_features) + tuple(reversed_orders))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=256)
    parser.add_argument("--eval-lifetimes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-every-updates", type=int, default=0,
                        help="emit compact in-epoch telemetry at this update cadence")
    parser.add_argument("--grounding-epochs", type=int, default=0,
                        help="epochs using stable first/last cues before novel cues")
    parser.add_argument("--last-epochs", type=int, default=0)
    parser.add_argument("--first-epochs", type=int, default=0)
    parser.add_argument("--evaluation-temporal-stage",
                        choices=("meta", "grounded", "first", "last"), default="meta")
    parser.add_argument("--order-routing", action="store_true")
    parser.add_argument("--write-binding", action="store_true")
    parser.add_argument("--event-binding", action="store_true")
    parser.add_argument("--event-binding-width", type=int, default=64)
    parser.add_argument("--event-binding-write-pairs", action="store_true",
                        help="also bind the final recurrent write state against each event snapshot")
    parser.add_argument("--event-binding-pairwise-transfer-checkpoint", type=Path)
    parser.add_argument("--event-binding-projection-transfer-checkpoint", type=Path)
    parser.add_argument("--event-binding-transfer-strength", type=float, default=0.0)
    parser.add_argument("--freeze-event-binding-transfer-strength", action="store_true")
    parser.add_argument("--preserve-raw-write", action="store_true")
    parser.add_argument("--preserve-first-raw-write", action="store_true",
                        help="retain the first support write as the raw sidecar")
    parser.add_argument("--latest-row-reader", action="store_true",
                        help="enable the optional trainable latest-memory-row channel")
    parser.add_argument("--latest-row-warmstart", type=Path,
                        help="cached key-projection bootstrap for the latest-row channel")
    parser.add_argument("--latest-row-answer-fusion", action="store_true",
                        help="add a zero-start answer head over controller + latest row")
    parser.add_argument("--latest-row-answer-gate", action="store_true",
                        help="learn a task-agnostic scalar gate for fusion logits")
    parser.add_argument("--latest-row-answer-entropy-threshold", type=float,
                        help="fixed entropy gate for answer fusion")
    parser.add_argument("--latest-row-answer-pairwise", action="store_true",
                        help="add a zero-start multiplicative query/memory answer head")
    parser.add_argument("--latest-row-answer-event-binding", action="store_true",
                        help="bind query event snapshots directly against latest memory")
    parser.add_argument("--latest-row-answer-event-linear", action="store_true",
                        help="use a linear audited readout over the event-binding features")
    parser.add_argument("--latest-row-answer-factorized-router",
                        action="store_true",
                        help="route between two learned query candidates using support memory")
    parser.add_argument("--latest-row-factorized-ood-threshold", type=float)
    parser.add_argument("--latest-row-fusion-only", action="store_true",
                        help="train only the latest-row answer-fusion head")
    parser.add_argument("--latest-row-pairwise-only", action="store_true",
                        help="train only the pairwise query/memory answer head")
    parser.add_argument("--latest-row-event-binding-only", action="store_true",
                        help="train only the query-event/latest-memory answer binder")
    parser.add_argument("--latest-row-factorized-router-only",
                        action="store_true",
                        help="train only the factorized answer router")
    parser.add_argument("--latest-row-gate-only", action="store_true",
                        help="train only the latest-row answer gate")
    parser.add_argument("--freeze-event-binding-module", action="store_true",
                        help="freeze all writer-binding parameters during reader diagnostics")
    parser.add_argument("--event-binding-warmstart", type=Path,
                        help="copy project/position weights from an audited pairwise snapshot probe")
    parser.add_argument("--router-only", action="store_true",
                        help="train only the new route, its gate, and the answer head")
    parser.add_argument("--writer-only", action="store_true",
                        help="train only write key/value/gate projections")
    parser.add_argument("--binder-only", action="store_true",
                        help="train only the generic pre-write binding module")
    parser.add_argument("--event-binder-only", action="store_true",
                        help="train only the generic event-snapshot write binder")
    parser.add_argument("--event-binder-reader-only", action="store_true",
                        help="train the event binder plus the minimal memory reader")
    parser.add_argument("--freeze-answer-head", action="store_true")
    parser.add_argument("--temporal-old-weight", type=float, default=2.0)
    parser.add_argument("--temporal-future-weight", type=float, default=1.0)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--temporal-only", action="store_true",
                        help="use only temporal lifetimes for a capacity diagnostic")
    parser.add_argument("--temporal-feedback-mode",
                        choices=("white-button", "color-button"),
                        default="white-button")
    parser.add_argument("--temporal-counterfactual-augmentation", action="store_true",
                        help="train on verifier-labeled order-reversed temporal queries")
    parser.add_argument("--repeat-training-lifetimes", action="store_true",
                        help="reuse identical lifetime seeds every epoch for an overfit test")
    parser.add_argument("--lifetime-seed-offset", type=int, default=0,
                        help="offset training lifetime IDs without changing model RNG")
    parser.add_argument("--lifetime-seed-offsets", default="",
                        help="comma-separated offsets cycled across training batches")
    parser.add_argument("--controller-learning-rate", type=float, default=2e-5)
    parser.add_argument("--reader-learning-rate", type=float, default=3e-4)
    parser.add_argument("--consolidator-learning-rate", type=float, default=1e-4)
    parser.add_argument("--write-rule-bootstrap", action="store_true",
                        help="train a disposable rule head on the actual raw write row")
    parser.add_argument("--write-rule-from-bound", action="store_true",
                        help="attach the disposable rule head directly to the event relation latent")
    parser.add_argument("--write-rule-pairwise-head", action="store_true",
                        help="use the proven pairwise diagnostic head on relation features")
    parser.add_argument("--write-rule-bootstrap-weight", type=float, default=1.0)
    parser.add_argument("--write-residual-penalty-weight", type=float, default=0.0)
    parser.add_argument("--read-action-bootstrap", action="store_true",
                        help="temporary action head on the post-read context")
    parser.add_argument("--read-action-bootstrap-weight", type=float, default=1.0)
    parser.add_argument("--read-rule-bootstrap", action="store_true",
                        help="temporary rule head on post-read context")
    parser.add_argument("--read-rule-bootstrap-weight", type=float, default=1.0)
    parser.add_argument("--resume-training-state", action="store_true",
                        help="resume optimizer/head/history from the controller checkpoint")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    lifetime_offsets = tuple(int(value) for value in args.lifetime_seed_offsets.split(",")
                            if value.strip())
    if not lifetime_offsets:
        lifetime_offsets = (args.lifetime_seed_offset,)
    transfer_paths = None
    if args.event_binding_pairwise_transfer_checkpoint or args.event_binding_projection_transfer_checkpoint:
        if not (args.event_binding_pairwise_transfer_checkpoint and
                args.event_binding_projection_transfer_checkpoint):
            raise ValueError("both pairwise and projection transfer checkpoints are required")
        transfer_paths = (str(args.event_binding_pairwise_transfer_checkpoint),
                          str(args.event_binding_projection_transfer_checkpoint))
    if (not args.temporal_only and
            args.train_lifetimes % (4 * args.batch_size)):
        raise ValueError("train lifetimes must contain complete 50/25/25 rehearsal cycles")
    if args.train_lifetimes % args.batch_size:
        raise ValueError("train lifetimes must be divisible by batch size")
    if (args.write_rule_bootstrap or args.write_rule_from_bound or args.write_rule_pairwise_head) and not args.event_binding:
        raise ValueError("--write-rule-bootstrap requires --event-binding")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.controller_checkpoint, map_location=device, weights_only=False)
    config = payload.get("controller_arguments", payload.get("arguments"))
    entropy_threshold = (args.latest_row_answer_entropy_threshold
                         if args.latest_row_answer_entropy_threshold is not None else
                         config.get("latest_row_answer_entropy_threshold"))
    model = NeuralComputerAgent(
        config["hidden"], config["workspace_slots"], config["heads"],
        config["thought_steps"], action_count=8, read_top_k=config["read_top_k"],
        order_routing=args.order_routing, write_binding=args.write_binding,
        event_binding=(args.event_binding or transfer_paths is not None),
        event_binding_width=args.event_binding_width,
        event_binding_write_pairs=args.event_binding_write_pairs,
        event_binding_pairwise_transfer=transfer_paths,
        latest_row_reader=args.latest_row_reader,
        latest_row_warmstart=(str(args.latest_row_warmstart)
                              if args.latest_row_warmstart else None),
        latest_row_answer_fusion=args.latest_row_answer_fusion,
        latest_row_answer_gate=args.latest_row_answer_gate,
        latest_row_answer_entropy_threshold=entropy_threshold,
        latest_row_answer_pairwise=args.latest_row_answer_pairwise,
        latest_row_answer_event_binding=args.latest_row_answer_event_binding,
        latest_row_answer_event_linear=args.latest_row_answer_event_linear,
        latest_row_answer_factorized_router=(
            args.latest_row_answer_factorized_router),
        latest_row_factorized_ood_threshold=(
            args.latest_row_factorized_ood_threshold)).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = ({key for key in incompatible.missing_keys
                        if key.startswith("answer_route")} if args.order_routing else set())
    if args.write_binding:
        allowed_missing.update(key for key in incompatible.missing_keys
                               if key.startswith("write_binding"))
    if args.event_binding or transfer_paths is not None:
        allowed_missing.update(key for key in incompatible.missing_keys
                               if key.startswith("event_binding_module"))
    if args.latest_row_reader:
        allowed_missing.update(key for key in incompatible.missing_keys
                               if key.startswith("latest_row_"))
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError(f"incompatible controller checkpoint: {incompatible}")
    if args.event_binding_warmstart:
        if not args.event_binding:
            raise ValueError("--event-binding-warmstart requires --event-binding")
        warm = torch.load(args.event_binding_warmstart, map_location="cpu", weights_only=False)
        warm_state = warm.get("model", warm)
        transferable = {key: value for key, value in warm_state.items()
                        if key.startswith(("project.", "positions"))}
        missing, unexpected = model.event_binding_module.load_state_dict(transferable, strict=False)
        if set(unexpected) or set(missing) != {key for key in model.event_binding_module.state_dict()
                                               if not key.startswith(("project.", "positions"))}:
            raise ValueError(f"invalid event-binder warm start: missing={missing}, unexpected={unexpected}")
    if transfer_paths is not None:
        # This tiny integration fork trains only the scalar transfer strength;
        # the audited relation/projection and generic binder stay frozen.
        for name, parameter in model.event_binding_module.named_parameters():
            parameter.requires_grad_(name == "pairwise_transfer.strength")
        model.event_binding_module.pairwise_transfer.strength.data.fill_(
            args.event_binding_transfer_strength)
        if args.freeze_event_binding_transfer_strength:
            model.event_binding_module.pairwise_transfer.strength.requires_grad_(False)
    compact_payload = torch.load(
        args.consolidator_checkpoint, map_location=device, weights_only=False)
    consolidator = LatentConsolidator(model.hidden, heads=config["heads"]).to(device)
    consolidator.load_state_dict(compact_payload["consolidator"])
    if sum((args.router_only, args.writer_only, args.binder_only,
            args.event_binder_only, args.event_binder_reader_only)) > 1:
        raise ValueError("restricted training modes are mutually exclusive")
    restricted_training = (args.router_only or args.writer_only or args.binder_only or
                           args.event_binder_only or args.event_binder_reader_only or
                           args.latest_row_fusion_only or args.latest_row_gate_only or
                           args.latest_row_pairwise_only or
                           args.latest_row_event_binding_only or
                           args.latest_row_factorized_router_only)
    if restricted_training:
        if not args.order_routing:
            if args.router_only:
                raise ValueError("--router-only requires --order-routing")
        if args.binder_only and not args.write_binding:
            raise ValueError("--binder-only requires --write-binding")
        if args.event_binder_only and not args.event_binding:
            raise ValueError("--event-binder-only requires --event-binding")
        if args.event_binder_reader_only and not args.event_binding:
            raise ValueError("--event-binder-reader-only requires --event-binding")
        if args.latest_row_fusion_only and not args.latest_row_answer_fusion:
            raise ValueError("--latest-row-fusion-only requires --latest-row-answer-fusion")
        if args.latest_row_gate_only and not (args.latest_row_answer_fusion and
                                              args.latest_row_answer_gate):
            raise ValueError("--latest-row-gate-only requires fusion and gate")
        if args.latest_row_pairwise_only and not args.latest_row_answer_pairwise:
            raise ValueError("--latest-row-pairwise-only requires pairwise fusion")
        if (args.latest_row_event_binding_only and
                not args.latest_row_answer_event_binding):
            raise ValueError(
                "--latest-row-event-binding-only requires event answer binding")
        if (args.latest_row_factorized_router_only and
                not args.latest_row_answer_factorized_router):
            raise ValueError(
                "--latest-row-factorized-router-only requires factorized routing")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            train_router = (args.router_only and
                            (name.startswith("answer_route") or
                             (name.startswith("answer_head") and
                              not args.freeze_answer_head)))
            train_writer = (args.writer_only and
                            name.startswith(("write_key", "write_value", "write_gate")))
            train_binder = args.binder_only and name.startswith("write_binding")
            train_event_binder = (args.event_binder_only and
                                  name.startswith("event_binding_module"))
            train_event_reader = (args.event_binder_reader_only and
                                  (name.startswith("event_binding_module") or
                                   name.startswith("latest_row_") or
                                   name.startswith(("read_query", "thought_cell",
                                                    "answer_head")) or
                                   name == "log_read_scale"))
            train_fusion = ((args.latest_row_fusion_only and
                             (name.startswith("latest_row_answer_fusion_head") or
                              name.startswith("latest_row_answer_gate_head"))) or
                            (args.latest_row_gate_only and
                             name.startswith("latest_row_answer_gate_head")))
            train_pairwise = (args.latest_row_pairwise_only and
                              name.startswith("latest_row_answer_pairwise_head"))
            train_answer_event_binding = (
                args.latest_row_event_binding_only and
                name.startswith(("latest_row_answer_event_project",
                                 "latest_row_answer_memory_project",
                                 "latest_row_answer_event_head")))
            train_factorized_router = (
                args.latest_row_factorized_router_only and
                name.startswith(("latest_row_factorized_router",
                                 "latest_row_factorized_strength")))
            if (train_router or train_writer or train_binder or train_event_binder or
                    train_event_reader or train_fusion or train_pairwise or
                    train_answer_event_binding or train_factorized_router):
                parameter.requires_grad_(True)
        for parameter in consolidator.parameters():
            parameter.requires_grad_(False)
        if args.event_binder_reader_only:
            binder_parameters = [parameter for name, parameter in model.named_parameters()
                                 if parameter.requires_grad and
                                 name.startswith("event_binding_module")]
            reader_parameters = [parameter for name, parameter in model.named_parameters()
                                 if parameter.requires_grad and
                                 not name.startswith("event_binding_module")]
            optimizer_groups = [
                {"params": binder_parameters, "lr": args.controller_learning_rate},
                {"params": reader_parameters, "lr": args.reader_learning_rate},
            ]
        else:
            optimizer_groups = [
                {"params": [parameter for parameter in model.parameters()
                            if parameter.requires_grad],
                 "lr": args.controller_learning_rate},
            ]
    if args.freeze_event_binding_module:
        for parameter in model.event_binding_module.parameters():
            parameter.requires_grad_(False)
    else:
        optimizer_groups = [
            {"params": model.parameters(), "lr": args.controller_learning_rate},
            {"params": consolidator.parameters(), "lr": args.consolidator_learning_rate},
        ]
    rule_head = None
    if args.write_rule_bootstrap or args.write_rule_from_bound or args.write_rule_pairwise_head:
        if args.write_rule_pairwise_head:
            rule_head = nn.Sequential(
                nn.Linear(args.event_binding_width * 9, args.event_binding_width * 3), nn.GELU(),
                nn.LayerNorm(args.event_binding_width * 3),
                nn.Linear(args.event_binding_width * 3, args.event_binding_width), nn.GELU(),
                nn.Linear(args.event_binding_width, 2)).to(device)
        else:
            rule_input_width = model.hidden if args.write_rule_from_bound else model.hidden * 2
            rule_head = nn.Sequential(
                nn.LayerNorm(rule_input_width),
                nn.Linear(rule_input_width, args.event_binding_width), nn.GELU(),
                nn.Linear(args.event_binding_width, 2)).to(device)
        optimizer_groups.append(
            {"params": rule_head.parameters(), "lr": args.controller_learning_rate})
    read_head = None
    if args.read_action_bootstrap:
        read_head = nn.Sequential(nn.LayerNorm(model.hidden),
                                  nn.Linear(model.hidden, 8)).to(device)
        optimizer_groups.append(
            {"params": read_head.parameters(), "lr": args.reader_learning_rate})
    read_rule_head = None
    if args.read_rule_bootstrap:
        read_rule_head = nn.Sequential(nn.LayerNorm(model.hidden),
                                       nn.Linear(model.hidden, 2)).to(device)
        optimizer_groups.append(
            {"params": read_rule_head.parameters(), "lr": args.reader_learning_rate})
    optimizer = torch.optim.AdamW(optimizer_groups)
    completed_epochs = 0
    previous_history = []
    previous_training_seconds = 0.0
    if args.resume_training_state:
        if "optimizer" not in payload or "completed_epochs" not in payload:
            raise ValueError("controller checkpoint has no resumable training state")
        if rule_head is not None:
            if payload.get("bootstrap_rule_head") is None:
                raise ValueError("checkpoint has no bootstrap rule head")
            rule_head.load_state_dict(payload["bootstrap_rule_head"])
        optimizer.load_state_dict(payload["optimizer"])
        completed_epochs = int(payload["completed_epochs"])
        previous_history = list(payload.get("history", []))
        previous_training_seconds = float(payload.get("training_seconds", 0.0))
        if completed_epochs >= args.epochs:
            raise ValueError("--epochs must exceed the checkpoint's completed epochs")
    saved_controller_config = dict(config)
    saved_controller_config["order_routing"] = args.order_routing
    saved_controller_config["write_binding"] = args.write_binding
    saved_controller_config["event_binding"] = (args.event_binding or transfer_paths is not None)
    saved_controller_config["event_binding_width"] = args.event_binding_width
    saved_controller_config["event_binding_write_pairs"] = args.event_binding_write_pairs
    saved_controller_config["write_rule_from_bound"] = args.write_rule_from_bound
    saved_controller_config["write_rule_pairwise_head"] = args.write_rule_pairwise_head
    saved_controller_config["event_binding_warmstart"] = (str(args.event_binding_warmstart)
                                                              if args.event_binding_warmstart else None)
    saved_controller_config["event_binding_pairwise_transfer"] = transfer_paths
    saved_controller_config["event_binding_transfer_strength"] = args.event_binding_transfer_strength
    saved_controller_config["preserve_raw_write"] = args.preserve_raw_write
    saved_controller_config["preserve_first_raw_write"] = args.preserve_first_raw_write
    saved_controller_config["latest_row_reader"] = args.latest_row_reader
    saved_controller_config["latest_row_warmstart"] = (str(args.latest_row_warmstart)
                                                          if args.latest_row_warmstart else None)
    saved_controller_config["latest_row_answer_fusion"] = args.latest_row_answer_fusion
    saved_controller_config["latest_row_answer_gate"] = args.latest_row_answer_gate
    saved_controller_config["latest_row_answer_entropy_threshold"] = entropy_threshold
    saved_controller_config["latest_row_answer_pairwise"] = args.latest_row_answer_pairwise
    saved_controller_config["latest_row_answer_event_binding"] = (
        args.latest_row_answer_event_binding)
    saved_controller_config["latest_row_answer_event_linear"] = (
        args.latest_row_answer_event_linear)
    saved_controller_config["latest_row_answer_factorized_router"] = (
        args.latest_row_answer_factorized_router)
    saved_controller_config["latest_row_factorized_ood_threshold"] = (
        args.latest_row_factorized_ood_threshold)

    history = previous_history
    last_completed_epoch = completed_epochs

    def save_checkpoint(path: Path, training_seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "consolidator": consolidator.state_dict(),
                    "bootstrap_rule_head": (rule_head.state_dict()
                                            if rule_head is not None else None),
                    "controller_arguments": saved_controller_config,
                    "optimizer": optimizer.state_dict(),
                    "completed_epochs": last_completed_epoch,
                    "history": history,
                    "training_seconds": training_seconds,
                    "arguments": vars(args)}, path)

    binding_residuals: list[float] = []
    binding_gradient_norms: list[float] = []
    binding_hook = None
    if args.event_binding or transfer_paths is not None:
        binding_hook = model.event_binding_module.register_forward_hook(
            lambda _module, inputs, output: binding_residuals.append(
                float((output.detach() - inputs[0].detach()).square().mean().sqrt())))
    cycle = (("temporal",) if args.temporal_only else
             ("temporal", "temporal", "spatial", "shape"))
    started = time.perf_counter()
    for epoch in range(completed_epochs, args.epochs):
        binding_residuals.clear()
        binding_gradient_norms.clear()
        model.train()
        consolidator.train()
        totals: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        latest_auxiliary_metrics: dict[str, float] = {}
        for offset in range(0, args.train_lifetimes, args.batch_size):
            update_in_epoch = offset // args.batch_size + 1
            residual_start = len(binding_residuals)
            primitive = cycle[(offset // args.batch_size) % len(cycle)]
            temporal_stage = None
            if epoch < args.last_epochs:
                temporal_stage = generate_temporal_last_lifetime
            elif epoch < args.last_epochs + args.first_epochs:
                temporal_stage = generate_temporal_first_lifetime
            elif epoch < args.last_epochs + args.first_epochs + args.grounding_epochs:
                temporal_stage = generate_temporal_grounding_lifetime
            generator = (temporal_stage if primitive == "temporal" and temporal_stage
                         else GENERATORS[primitive])
            offset_index = (offset // args.batch_size) % len(lifetime_offsets)
            epoch_seed = (lifetime_offsets[offset_index] +
                          (0 if args.repeat_training_lifetimes else epoch * args.train_lifetimes))
            generator_kwargs = {"query_count": args.query_count}
            if generator is generate_temporal_attention_lifetime:
                generator_kwargs["feedback_mode"] = args.temporal_feedback_mode
            lifetimes = [generator(epoch_seed + offset + index, **generator_kwargs)
                         for index in range(args.batch_size)]
            if args.temporal_counterfactual_augmentation and primitive == "temporal":
                lifetimes = [_add_temporal_counterfactual_queries(item)
                             for item in lifetimes]
            rule_targets = None
            auxiliary = None
            auxiliary_accuracies = []
            if (rule_head is not None or read_rule_head is not None) and primitive == "temporal":
                rule_targets = torch.tensor(
                    [item.rule for item in lifetimes], device=device, dtype=torch.long)
                if rule_head is not None:
                    def auxiliary(output):
                        if args.write_rule_pairwise_head:
                            logits = rule_head(model.event_binding_module.last_relation_features)
                        elif args.write_rule_from_bound:
                            bound = model.event_binding_module.last_bound
                            logits = rule_head(bound)
                        else:
                            raw_write = torch.cat((output.write_keys, output.write_values), dim=-1)
                            logits = rule_head(raw_write)
                        auxiliary_accuracies.append(float(
                            (logits.detach().argmax(dim=-1) == rule_targets).float().mean()))
                        return nn.functional.cross_entropy(logits, rule_targets)
            read_auxiliary = None
            if read_head is not None or (read_rule_head is not None and rule_targets is not None):
                def read_auxiliary(output, targets):
                    total = torch.zeros((), device=device)
                    if read_head is not None:
                        total = total + args.read_action_bootstrap_weight * nn.functional.cross_entropy(
                            read_head(output.read_context), targets)
                    if read_rule_head is not None and rule_targets is not None:
                        total = total + args.read_rule_bootstrap_weight * nn.functional.cross_entropy(
                            read_rule_head(output.read_context), rule_targets)
                    return total
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_compaction_batch(
                model, consolidator, lifetimes, device, train=True, train_model=True,
                preserve_raw_write=args.preserve_raw_write,
                preserve_first_raw_write=args.preserve_first_raw_write,
                old_loss_weight=(args.temporal_old_weight if primitive == "temporal" else 2.0),
                future_loss_weight=(args.temporal_future_weight
                                    if primitive == "temporal" else 1.0),
                write_auxiliary_loss=auxiliary,
                write_auxiliary_weight=(args.write_rule_bootstrap_weight
                                        if primitive == "temporal" else 0.0),
                read_auxiliary_loss=read_auxiliary,
                write_residual_penalty_weight=args.write_residual_penalty_weight)
            if auxiliary_accuracies:
                metrics["write_rule_accuracy"] = (
                    sum(auxiliary_accuracies) / len(auxiliary_accuracies))
            for key in ("write_auxiliary_loss", "write_rule_accuracy"):
                if key in metrics:
                    latest_auxiliary_metrics[key] = metrics[key]
            loss.backward()
            if args.event_binding:
                squared_norm = sum((
                    parameter.grad.detach().float().square().sum()
                    for parameter in model.event_binding_module.parameters()
                    if parameter.grad is not None), torch.zeros((), device=device))
                binding_gradient_norms.append(float(squared_norm.sqrt()))
                current_gradient_norm = float(squared_norm.sqrt())
            else:
                current_gradient_norm = 0.0
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            if not restricted_training:
                nn.utils.clip_grad_norm_(consolidator.parameters(), 1.0)
            optimizer.step()
            if (args.log_every_updates and
                    update_in_epoch % args.log_every_updates == 0):
                recent_residuals = binding_residuals[residual_start:]
                progress = {
                    "epoch": epoch + 1,
                    "update_in_epoch": update_in_epoch,
                    "updates_in_epoch": args.train_lifetimes // args.batch_size,
                    "primitive": primitive,
                    "loss": float(loss.detach()),
                    "event_binding_gradient_norm": current_gradient_norm,
                    "event_binding_residual_rms": (
                        sum(recent_residuals) / max(1, len(recent_residuals))),
                }
                for key in ("write_auxiliary_loss", "write_rule_accuracy",
                            "compact_few_shot_auc", "full_few_shot_auc"):
                    if key in metrics:
                        progress[key] = metrics[key]
                progress.update(latest_auxiliary_metrics)
                print(json.dumps({"progress": progress}, sort_keys=True), flush=True)
            bucket = totals.setdefault(primitive, {})
            for key, value in {"loss": float(loss.detach()), **metrics}.items():
                bucket[key] = bucket.get(key, 0.0) + value
            counts[primitive] = counts.get(primitive, 0) + 1
        row = {primitive: {key: value / counts[primitive]
                           for key, value in values.items()}
               for primitive, values in totals.items()}
        row["epoch"] = epoch + 1
        if args.event_binding:
            row["event_binding_diagnostics"] = {
                "gradient_norm": (sum(binding_gradient_norms) /
                                  max(1, len(binding_gradient_norms))),
                "residual_rms": (sum(binding_residuals) /
                                 max(1, len(binding_residuals))),
            }
        history.append(row)
        last_completed_epoch = epoch + 1
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            periodic = args.checkpoint.with_name(
                f"{args.checkpoint.stem}.epoch_{epoch + 1:04d}{args.checkpoint.suffix}")
            save_checkpoint(
                periodic, previous_training_seconds + time.perf_counter() - started)
    if binding_hook is not None:
        binding_hook.remove()
    model.eval()
    consolidator.eval()
    evaluation_generators = dict(GENERATORS)
    evaluation_generators["temporal"] = {
        "meta": partial(generate_temporal_attention_lifetime,
                        feedback_mode=args.temporal_feedback_mode),
        "grounded": generate_temporal_grounding_lifetime,
        "first": generate_temporal_first_lifetime,
        "last": generate_temporal_last_lifetime,
    }[args.evaluation_temporal_stage]
    evaluation = {
        primitive: evaluate(
            model, consolidator, device, samples=args.eval_lifetimes,
            batch_size=args.batch_size,
            seed=4_000_000 + args.seed * 10_000,
            query_count=args.query_count, generator=generator)
        for primitive, generator in evaluation_generators.items()
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    elapsed_training_seconds = previous_training_seconds + time.perf_counter() - started
    save_checkpoint(args.checkpoint, elapsed_training_seconds)
    report = {
        "schema": "forward-transfer-joint-adapter-v1",
        "sensory_only": True,
        "parameters": parameter_count(model),
        "history": history,
        "evaluation": evaluation,
        "training_seconds": elapsed_training_seconds,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation": evaluation}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
