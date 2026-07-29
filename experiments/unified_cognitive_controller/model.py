"""One recurrent controller coordinating sensory encoding and latent memory."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .environment import ACTIONS, NULL_ACTION


def full_memory_usage_features(
        base_features: torch.Tensor, queries: torch.Tensor,
        keys: torch.Tensor, usage: torch.Tensor,
        *, retained_rows: int = 4) -> torch.Tensor:
    """Append sorted per-row content and usage evidence to legacy features."""
    if retained_rows < 1:
        raise ValueError("retained_rows must be positive")
    if base_features.ndim != 2 or base_features.shape[1] != 4:
        raise ValueError("base features must have shape [batch, 4]")
    if keys.ndim != 3 or usage.shape != keys.shape[:2]:
        raise ValueError("keys/usage shapes do not describe row banks")
    normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
    normalized_keys = torch.nn.functional.normalize(keys, dim=-1)
    cosine = torch.einsum(
        "bw,bkw->bk", normalized_queries, normalized_keys)
    order = cosine.argsort(dim=-1, descending=True)
    cosine = torch.gather(cosine, 1, order)
    sorted_usage = torch.gather(usage, 1, order)
    missing = retained_rows - cosine.shape[1]
    if missing > 0:
        cosine = torch.nn.functional.pad(
            cosine, (0, missing), value=-1.0)
        sorted_usage = torch.nn.functional.pad(
            sorted_usage, (0, missing), value=0.0)
    return torch.cat((
        base_features,
        cosine[:, :retained_rows],
        sorted_usage[:, :retained_rows]), dim=-1)


class VisionEventEncoder(nn.Module):
    """Small modality encoder; it receives pixels and emits one event latent."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(width),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.network(frames)


@dataclass
class ControllerState:
    hidden: torch.Tensor
    workspace: torch.Tensor

    def detach(self) -> "ControllerState":
        return ControllerState(self.hidden.detach(), self.workspace.detach())


@dataclass
class ControllerOutput:
    logits: torch.Tensor
    intention: torch.Tensor
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    memory_write_strength: torch.Tensor
    workspace_read: torch.Tensor
    # Per-slot opening of the successor residuals, shape [batch, slots]. It is
    # exposed so a rung can measure and price how far its own new slot fires
    # outside the events it was added for. Empty when no slot exists.
    skill_adapter_openings: torch.Tensor | None = None
    # Norm of the perturbation each successor slot actually adds to the
    # intention, shape [batch, slots]. The opening alone understates this: a
    # nearly shut gate on a large residual still moves the answer, so this is
    # the quantity a locality price has to act on.
    skill_adapter_residual_norms: torch.Tensor | None = None


class UnifiedCognitiveController(nn.Module):
    """A single task-agnostic controller with a differentiable RAM workspace.

    There are no primitive-specific heads.  The same encoder, recurrent cell,
    workspace operations, latent intention, and actuator adapter process every
    trial. Persistent-memory keys and values are exposed for the later disk
    rung, but disk writes are deliberately disabled in the first fast-learning
    experiment.
    """

    def __init__(
            self, width: int = 96, workspace_slots: int = 8,
            intention_width: int = 24,
            adaptive_memory_read: bool = False,
            adaptive_memory_read_hidden: int = 0,
            adaptive_memory_replace: bool = False,
            adaptive_memory_replace_hidden: int = 8,
            adaptive_memory_replace_features: int = 5,
            adaptive_memory_usage_prior: bool = False,
            adaptive_memory_usage_prior_hidden: int = 0,
            adaptive_memory_usage_prior_residual_hidden: int = 0,
            adaptive_memory_usage_prior_residual_features: int = 4,
            adaptive_memory_usage_prior_proposer_hidden: int = 0,
            adaptive_memory_equivalence_hidden: int = 0,
            adaptive_memory_equivalence_calibration: bool = False,
            adaptive_representative_read_hidden: int = 0,
            adaptive_representative_read_threshold: float = 0.01,
            relation_adapter_width: int = 0,
            relation_adapter_gated: bool = False,
            action_adapter_width: int = 0,
            action_adapter_gated: bool = False,
            skill_adapter_widths: tuple[int, ...] = (),
            skill_adapter_gate_mode: str = "sigmoid",
            skill_adapter_gate_hidden: int = 0,
            skill_adapter_reads_prior: bool = False,
            skill_adapter_legacy_read_from: int | None = None,
            skill_adapter_reads_prior_from: int | None = None,
            skill_adapter_read_bottleneck: int = 0,
            skill_adapter_prior_read_limit: int = 0) -> None:
        super().__init__()
        if skill_adapter_gate_mode not in ("sigmoid", "relu"):
            raise ValueError(
                "skill adapter gate mode must be sigmoid or relu")
        if (
                width < 16 or workspace_slots < 1 or intention_width < 2
                or adaptive_memory_read_hidden < 0
                or adaptive_memory_replace_hidden < 1
                or adaptive_memory_replace_features < 5
                or adaptive_memory_usage_prior_hidden < 0
                or adaptive_memory_usage_prior_residual_hidden < 0
                or adaptive_memory_usage_prior_residual_features < 4
                or adaptive_memory_usage_prior_proposer_hidden < 0
                or adaptive_memory_equivalence_hidden < 0
                or adaptive_representative_read_hidden < 0
                or not 0.0 < adaptive_representative_read_threshold < 1.0
                or relation_adapter_width < 0
                or action_adapter_width < 0
                or skill_adapter_prior_read_limit < 0
                or any(value < 1 for value in skill_adapter_widths)):
            raise ValueError("controller dimensions are too small")
        self.width = width
        self.workspace_slots = workspace_slots
        self.intention_width = intention_width
        self.adaptive_memory_read = adaptive_memory_read
        self.adaptive_memory_read_hidden = adaptive_memory_read_hidden
        self.adaptive_memory_replace = adaptive_memory_replace
        self.adaptive_memory_replace_hidden = adaptive_memory_replace_hidden
        self.adaptive_memory_replace_features = adaptive_memory_replace_features
        self.adaptive_memory_usage_prior = adaptive_memory_usage_prior
        self.adaptive_memory_usage_prior_hidden = (
            adaptive_memory_usage_prior_hidden)
        self.adaptive_memory_usage_prior_residual_hidden = (
            adaptive_memory_usage_prior_residual_hidden)
        self.adaptive_memory_usage_prior_residual_features = (
            adaptive_memory_usage_prior_residual_features)
        self.adaptive_memory_usage_prior_proposer_hidden = (
            adaptive_memory_usage_prior_proposer_hidden)
        self.adaptive_memory_equivalence_hidden = (
            adaptive_memory_equivalence_hidden)
        self.adaptive_memory_equivalence_calibration = (
            adaptive_memory_equivalence_calibration)
        self.adaptive_representative_read_hidden = (
            adaptive_representative_read_hidden)
        self.adaptive_representative_read_threshold = (
            adaptive_representative_read_threshold)
        self.relation_adapter_width = relation_adapter_width
        self.relation_adapter_gated = relation_adapter_gated
        self.action_adapter_width = action_adapter_width
        self.action_adapter_gated = action_adapter_gated
        self.skill_adapter_widths = tuple(skill_adapter_widths)
        self.skill_adapter_gate_mode = skill_adapter_gate_mode
        self.skill_adapter_gate_hidden = skill_adapter_gate_hidden
        # Whether a slot may read what earlier slots computed, separately from
        # whether those slots write to the answer. An exactly shut gate makes an
        # earlier slot silent on this event -- which is what removes
        # interference, and also what makes deeper ancestry invisible to the
        # next slot, since its features are then bit-identical whatever came
        # before. Reading a prior slot's pre-gate hidden layer restores the
        # information without restoring the disturbance.
        self.skill_adapter_reads_prior = skill_adapter_reads_prior
        # Ablation for the read path. A reading slot has a wider first layer
        # than a non-reading one, so a speedup could be extra capacity rather
        # than inherited information. Zeroing the prior read keeps the shape and
        # the parameter count and removes only the content, which is the
        # comparison that separates the two.
        self.skill_adapter_ablate_prior_read = False
        # Index of the first slot allowed to read the two legacy adapters.
        # Rungs two and three consolidated into those, so they are ancestry a
        # later slot cannot otherwise see. It is an index rather than a flag so
        # that slots written before this existed keep their original input
        # width and their checkpoints still load.
        self.skill_adapter_legacy_read_from = skill_adapter_legacy_read_from
        # Index of the first slot that reads earlier slots. Without this, turning
        # reads on widens EVERY slot's input, and a checkpoint holding two or
        # more slots trained without reads will not load at all. Making it an
        # index means only the slot a rung adds takes the wider input, exactly
        # as for the legacy reads.
        self.skill_adapter_reads_prior_from = skill_adapter_reads_prior_from
        # Width to compress everything a slot reads down to. Reading one prior
        # slot (64 extra inputs) helped; reading two, or the legacy pair (128),
        # hurt, and hurt absolute learning badly in the legacy case. A slot has
        # only its own hidden width to work with, so a wide read appears to
        # dilute rather than inform. Zero keeps the raw concatenation.
        self.skill_adapter_read_bottleneck = skill_adapter_read_bottleneck
        # Zero reads every earlier slot. A positive value keeps only that many
        # immediately preceding slots. The first readable ancestor helped
        # strongly, while exposing a second did not improve the learning gain;
        # this selector tests whether local reuse can compound without carrying
        # every older intermediate representation into every new rung.
        self.skill_adapter_prior_read_limit = skill_adapter_prior_read_limit
        # Training-time leak below the rectifier's knee. A rectified gate that
        # shuts everywhere before it has learned where to open has no gradient
        # left and stays shut forever; a small leak keeps that recoverable. It
        # is a schedule, not architecture, so it is a plain runtime attribute
        # that no checkpoint stores, and it must be returned to zero before any
        # measurement: only an exact zero leaves an inherited skill untouched.
        self.skill_adapter_gate_leak = 0.0
        self.vision = VisionEventEncoder(width)
        self.action_embedding = nn.Embedding(ACTIONS + 1, width // 4)
        feedback_width = width // 4
        self.feedback_encoder = nn.Sequential(
            nn.Linear(2, feedback_width), nn.Tanh())
        self.read_query = nn.Linear(width * 2, width)
        controller_input = width * 3 + width // 4 + feedback_width
        self.controller = nn.GRUCell(controller_input, width)
        self.write_gate = nn.Linear(width * 3, 1)
        self.write_query = nn.Linear(width * 3, width)
        self.write_value = nn.Sequential(
            nn.Linear(width * 3, width), nn.Tanh())
        self.intention = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, intention_width),
            nn.Tanh(),
        )
        # This is the replaceable device/protocol adapter. The controller emits
        # an abstract intention before this final mapping.
        self.actuator = nn.Linear(intention_width, ACTIONS)
        # Optional generic answer-path relation adapter. It combines the
        # controller state available before a query with that query's sensory
        # event; it has no task, context, or action-label input. A zero final
        # projection makes insertion exactly behavior-preserving.
        self.relation_adapter = (
            nn.Sequential(
                nn.Linear(width * 2, relation_adapter_width),
                nn.GELU(),
                nn.Linear(relation_adapter_width, intention_width),
            )
            if relation_adapter_width else None)
        if self.relation_adapter is not None:
            nn.init.zeros_(self.relation_adapter[-1].weight)
            nn.init.zeros_(self.relation_adapter[-1].bias)
        self.relation_adapter_gate = (
            nn.Linear(width * 2, 1)
            if relation_adapter_width and relation_adapter_gated else None)
        if self.relation_adapter_gate is not None:
            nn.init.zeros_(self.relation_adapter_gate.weight)
            nn.init.constant_(self.relation_adapter_gate.bias, -2.0)
        self.action_adapter = (
            nn.Sequential(
                nn.Linear(width * 2, action_adapter_width),
                nn.GELU(),
                nn.Linear(action_adapter_width, ACTIONS),
            )
            if action_adapter_width else None)
        if self.action_adapter is not None:
            nn.init.zeros_(self.action_adapter[-1].weight)
            nn.init.zeros_(self.action_adapter[-1].bias)
        self.action_adapter_gate = (
            nn.Linear(width * 2, 1) if action_adapter_width and action_adapter_gated else None)
        if self.action_adapter_gate is not None:
            nn.init.zeros_(self.action_adapter_gate.weight)
            # The residual output itself is exactly zero at insertion, so a
            # moderately open gate preserves behavior while avoiding the
            # near-zero gradients caused by a saturated closed gate.
            nn.init.constant_(self.action_adapter_gate.bias, -2.0)
        # Indexable successor slots for later primitives. The two adapters
        # above were each claimed by one earlier rung; this stack lets rung N
        # add exactly one fresh zero-output residual without renaming or
        # reshaping anything an already promoted checkpoint stored. An empty
        # stack contributes no state, so older checkpoints load unchanged.
        self.skill_adapters = nn.ModuleList()
        self.skill_adapter_gates = nn.ModuleList()
        self.skill_adapter_read_projections = nn.ModuleList()
        legacy_read_width = (
            (relation_adapter_width if relation_adapter_width else 0)
            + (action_adapter_width if action_adapter_width else 0))
        prior_read_width = 0
        for slot_index, slot_width in enumerate(self.skill_adapter_widths):
            reads_legacy = (
                skill_adapter_legacy_read_from is not None
                and slot_index >= skill_adapter_legacy_read_from)
            reads_prior = skill_adapter_reads_prior and (
                skill_adapter_reads_prior_from is None
                or slot_index >= skill_adapter_reads_prior_from)
            selected_prior_width = prior_read_width
            if skill_adapter_prior_read_limit:
                selected_prior_width = sum(
                    self.skill_adapter_widths[
                        max(0, slot_index - skill_adapter_prior_read_limit):
                        slot_index])
            raw_read = (
                (selected_prior_width if reads_prior else 0)
                + (legacy_read_width if reads_legacy else 0))
            if raw_read and skill_adapter_read_bottleneck:
                slot_input = width * 2 + skill_adapter_read_bottleneck
            else:
                slot_input = width * 2 + raw_read
            adapter = nn.Sequential(
                nn.Linear(slot_input, slot_width),
                nn.GELU(),
                nn.Linear(slot_width, intention_width),
            )
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            self.skill_adapters.append(adapter)
            self.skill_adapter_read_projections.append(
                nn.Linear(raw_read, skill_adapter_read_bottleneck)
                if raw_read and skill_adapter_read_bottleneck
                else nn.Identity())
            prior_read_width += slot_width
            # A linear gate has to separate this slot's own events from every
            # other skill's with one hyperplane. A hidden layer lets the
            # boundary bend, which is what decides how cleanly a slot can be
            # both fully available to its operation and fully absent elsewhere.
            if skill_adapter_gate_hidden:
                gate = nn.Sequential(
                    nn.Linear(slot_input, skill_adapter_gate_hidden),
                    nn.GELU(),
                    nn.Linear(skill_adapter_gate_hidden, 1),
                )
                output_layer = gate[-1]
            else:
                gate = nn.Linear(slot_input, 1)
                output_layer = gate
            nn.init.zeros_(output_layer.weight)
            # A sigmoid gate can never be exactly shut, so a slot always
            # perturbs every event it sees a little. The rectified gate can
            # reach exact zero and therefore be genuinely inert outside the
            # events it was added for. Either way the zero-initialized output
            # layer makes insertion exactly behavior-preserving; the positive
            # rectified bias only keeps the gate's own gradient alive.
            nn.init.constant_(
                output_layer.bias,
                1.0 if skill_adapter_gate_mode == "relu" else -2.0)
            self.skill_adapter_gates.append(gate)
        self.memory_key = nn.Linear(width * 2, width)
        self.memory_value = nn.Linear(width * 2, width)
        self.memory_write = nn.Linear(width * 2, 1)
        if not adaptive_memory_read:
            self.memory_read_gate = None
        elif adaptive_memory_read_hidden == 0:
            self.memory_read_gate = nn.Linear(4, 1)
        else:
            self.memory_read_gate = nn.Sequential(
                nn.Linear(4, adaptive_memory_read_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_read_hidden, 1),
            )
        self.memory_replacement_gate = (
            nn.Sequential(
                nn.Linear(5, adaptive_memory_replace_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_replace_hidden, 1),
            )
            if adaptive_memory_replace else None)
        self.memory_replacement_extra_gate = (
            nn.Linear(
                adaptive_memory_replace_features - 5, 1, bias=False)
            if (
                adaptive_memory_replace
                and adaptive_memory_replace_features > 5)
            else None)
        if self.memory_replacement_extra_gate is not None:
            nn.init.zeros_(self.memory_replacement_extra_gate.weight)
        self.memory_usage_prior_scale = (
            nn.Parameter(torch.ones(()))
            if adaptive_memory_usage_prior else None)
        self.memory_usage_prior_policy = (
            nn.Sequential(
                nn.Linear(4, adaptive_memory_usage_prior_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_usage_prior_hidden, 1),
            )
            if adaptive_memory_usage_prior_hidden > 0 else None)
        if self.memory_usage_prior_policy is not None:
            output = self.memory_usage_prior_policy[-1]
            nn.init.zeros_(output.weight)
            # Hard inference remains exactly content-first (< 0.5), while
            # stochastic training still explores the usage-prior action.
            nn.init.constant_(output.bias, -2.0)
        self.memory_usage_prior_residual = (
            nn.Sequential(
                nn.Linear(
                    adaptive_memory_usage_prior_residual_features,
                    adaptive_memory_usage_prior_residual_hidden),
                nn.GELU(),
                nn.Linear(
                    adaptive_memory_usage_prior_residual_hidden, 1),
            )
            if adaptive_memory_usage_prior_residual_hidden > 0 else None)
        if self.memory_usage_prior_residual is not None:
            # Adding the branch is bit-identical until verified experience
            # earns a departure from the inherited retrieval function.
            nn.init.zeros_(self.memory_usage_prior_residual[-1].weight)
            nn.init.zeros_(self.memory_usage_prior_residual[-1].bias)
        self.memory_usage_prior_proposer = (
            nn.Sequential(
                nn.Linear(7, adaptive_memory_usage_prior_proposer_hidden),
                nn.GELU(),
                nn.Linear(
                    adaptive_memory_usage_prior_proposer_hidden, 5),
            )
            if adaptive_memory_usage_prior_proposer_hidden > 0 else None)
        if self.memory_usage_prior_proposer is not None:
            # Begin from an unbiased distribution over the four generic
            # intervals.  Verified candidate outcomes, rather than random
            # initialization, should decide which proposal becomes useful.
            nn.init.zeros_(self.memory_usage_prior_proposer[-1].weight)
            nn.init.zeros_(self.memory_usage_prior_proposer[-1].bias)
            # The fifth output opens the proposal path.  Its zero value makes
            # the inherited probability bit-identical while the
            # straight-through gate below retains a live derivative.
        self.memory_equivalence_selector = (
            nn.Sequential(
                nn.Linear(width * 4, adaptive_memory_equivalence_hidden),
                nn.GELU(),
                nn.Linear(adaptive_memory_equivalence_hidden, 1),
            )
            if adaptive_memory_equivalence_hidden > 0 else None)
        self.memory_equivalence_opening = (
            nn.Parameter(torch.zeros(()))
            if adaptive_memory_equivalence_hidden > 0 else None)
        if self.memory_equivalence_selector is not None:
            # The relation itself starts unbiased, and the scalar opening makes
            # adding this path exactly behavior preserving.  Candidate outcome
            # credit can shape the selector before the opening earns influence.
            nn.init.zeros_(self.memory_equivalence_selector[-1].weight)
            nn.init.zeros_(self.memory_equivalence_selector[-1].bias)
        self.memory_equivalence_logit_scale = (
            nn.Parameter(torch.ones(()))
            if adaptive_memory_equivalence_calibration else None)
        self.memory_equivalence_logit_bias = (
            nn.Parameter(torch.zeros(()))
            if adaptive_memory_equivalence_calibration else None)
        representative_features = width * 5 + 4
        self.representative_read_critic = (
            nn.Sequential(
                nn.Linear(
                    representative_features,
                    adaptive_representative_read_hidden),
                nn.GELU(),
                nn.Linear(
                    adaptive_representative_read_hidden,
                    max(adaptive_representative_read_hidden // 4, 4)),
                nn.GELU(),
                nn.Linear(
                    max(adaptive_representative_read_hidden // 4, 4), 1),
            )
            if adaptive_representative_read_hidden > 0 else None)
        self.register_buffer(
            "representative_read_feature_mean",
            torch.zeros(representative_features)
            if adaptive_representative_read_hidden > 0 else None)
        self.register_buffer(
            "representative_read_feature_scale",
            torch.ones(representative_features)
            if adaptive_representative_read_hidden > 0 else None)

    def effective_memory_usage_prior_scale(self) -> torch.Tensor:
        """Return the task-agnostic nonnegative retrieval-prior strength."""
        if self.memory_usage_prior_scale is None:
            return torch.ones(
                (), device=self.memory_key.weight.device,
                dtype=self.memory_key.weight.dtype)
        return self.memory_usage_prior_scale.clamp(0.0, 1.0)

    def memory_usage_prior_logits(
            self, features: torch.Tensor) -> torch.Tensor:
        """Return inherited retrieval logits plus any learned residual."""
        if self.memory_usage_prior_policy is None:
            raise RuntimeError("conditional memory usage prior is not enabled")
        if features.ndim != 2 or features.shape[1] < 4:
            raise ValueError(
                "usage-prior features must have shape [queries, >=4]")
        logits = self.memory_usage_prior_policy(
            features[:, :4]).squeeze(-1)
        if self.memory_usage_prior_residual is not None:
            expected = self.adaptive_memory_usage_prior_residual_features
            residual_features = features[:, :expected]
            if residual_features.shape[1] < expected:
                residual_features = torch.nn.functional.pad(
                    residual_features,
                    (0, expected - residual_features.shape[1]))
            logits = (
                logits
                + self.memory_usage_prior_residual(
                    residual_features).squeeze(-1))
        return logits

    def memory_usage_prior_probability(
            self, features: torch.Tensor) -> torch.Tensor:
        """Choose whether verified usage should influence each memory query."""
        probability = torch.sigmoid(self.memory_usage_prior_logits(features))
        if self.memory_usage_prior_proposer is None:
            return probability
        proposal_features, candidates = (
            self.memory_usage_prior_proposal_features(features))
        proposal_outputs = self.memory_usage_prior_proposer(
            proposal_features)
        proposal = (
            proposal_outputs[:, :4].softmax(dim=-1) * candidates
        ).sum(dim=-1)
        # Forward behavior is a bounded, exactly zero no-op at insertion.
        # The straight-through derivative remains live outside the interval,
        # so one negative exploratory update cannot permanently kill the new
        # path before verifier experience has had a chance to shape it.
        raw_opening = proposal_outputs[:, 4]
        bounded_opening = raw_opening.clamp(0.0, 1.0)
        opening = (
            raw_opening + (bounded_opening - raw_opening).detach())
        return probability + opening * (proposal - probability)

    @staticmethod
    def memory_usage_prior_proposal_features(
            features: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Describe the four generic intervals where row rank is constant."""
        if features.ndim != 2 or features.shape[1] < 12:
            raise ValueError(
                "relational usage proposer requires twelve row features")
        cosine = features[:, 4:8]
        log_usage = features[:, 8:12].clamp_min(1e-6).log()
        slope_gap = (log_usage[:, 1:] - log_usage[:, :-1]).clamp_min(1e-5)
        crossings = (
            (cosine[:, :-1] - cosine[:, 1:]) / slope_gap
        ).clamp(0.0, 1.0)
        candidates = torch.stack((
            crossings[:, 0] / 2.0,
            (crossings[:, 0] + crossings[:, 1]) / 2.0,
            (crossings[:, 1] + crossings[:, 2]) / 2.0,
            (crossings[:, 2] + 1.0) / 2.0,
        ), dim=-1).clamp(0.001, 0.999)
        return torch.cat((crossings, log_usage), dim=-1), candidates

    def memory_usage_prior_candidates(
            self, features: torch.Tensor) -> torch.Tensor:
        """Return task-agnostic scales that sample every rank interval."""
        return self.memory_usage_prior_proposal_features(features)[1]

    def memory_equivalence_logits(
            self, probe_values: torch.Tensor,
            sorted_row_values: torch.Tensor) -> torch.Tensor:
        """Score stored values against a fresh, learner-visible memory value.

        Rows must use the same descending-content order as the generic
        retrieval candidates.  The shared scorer has no row identity and sees
        only learned latents; verifier-private rules never enter this path.
        """
        if self.memory_equivalence_selector is None:
            raise RuntimeError("memory equivalence selector is not enabled")
        if (
                probe_values.ndim != 2
                or sorted_row_values.ndim != 3
                or sorted_row_values.shape[0] != probe_values.shape[0]
                or sorted_row_values.shape[1] < 1
                or probe_values.shape[1] != self.width
                or sorted_row_values.shape[2] != self.width):
            raise ValueError(
                "equivalence inputs must have shapes [batch, width] and "
                "[batch, rows, width]")
        probe = probe_values.unsqueeze(1).expand(
            -1, sorted_row_values.shape[1], -1)
        pair = torch.cat((
            probe,
            sorted_row_values,
            (probe - sorted_row_values).abs(),
            probe * sorted_row_values,
        ), dim=-1)
        return self.memory_equivalence_selector(pair).squeeze(-1)

    def calibrated_memory_equivalence_logits(
            self, probe_values: torch.Tensor,
            row_values: torch.Tensor) -> torch.Tensor:
        """Return absolute same-behavior evidence for merge/store decisions."""
        if (
                self.memory_equivalence_logit_scale is None
                or self.memory_equivalence_logit_bias is None):
            raise RuntimeError("memory equivalence calibration is not enabled")
        logits = self.memory_equivalence_logits(probe_values, row_values)
        # A nonnegative scale preserves the already-audited relation ordering;
        # the learned bias supplies the absolute equivalence threshold.
        scale = self.memory_equivalence_logit_scale.clamp_min(0.0)
        return logits * scale + self.memory_equivalence_logit_bias

    def representative_deep_read_probability(
            self, features: torch.Tensor) -> torch.Tensor:
        """Predict whether consulting extra within-class rows will help."""
        if self.representative_read_critic is None:
            raise RuntimeError(
                "adaptive representative reading is not enabled")
        expected = self.width * 5 + 4
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(
                f"representative read features must have shape "
                f"[batch, {expected}]")
        assert self.representative_read_feature_mean is not None
        assert self.representative_read_feature_scale is not None
        normalized = (
            (features - self.representative_read_feature_mean)
            / self.representative_read_feature_scale.clamp_min(1e-4))
        return torch.sigmoid(
            self.representative_read_critic(normalized).squeeze(-1))

    def memory_equivalence_probability(
            self, features: torch.Tensor, probe_values: torch.Tensor,
            sorted_row_values: torch.Tensor) -> torch.Tensor:
        """Retrieve through a hard relational choice with soft credit.

        The forward pass chooses one physical rank interval, avoiding invalid
        averages between disconnected but behaviorally equivalent intervals.
        The straight-through one-hot retains the softmax derivative.
        """
        if self.memory_equivalence_opening is None:
            raise RuntimeError("memory equivalence selector is not enabled")
        inherited = self.memory_usage_prior_probability(features)
        logits = self.memory_equivalence_logits(
            probe_values, sorted_row_values)
        if logits.shape[1] != 4:
            raise ValueError(
                "retrieval probability requires exactly four stored rows")
        soft = logits.softmax(dim=-1)
        hard = torch.nn.functional.one_hot(
            logits.argmax(dim=-1), num_classes=4).to(soft.dtype)
        choice = soft + (hard - soft).detach()
        candidates = self.memory_usage_prior_candidates(features)
        proposal = (choice * candidates).sum(dim=-1)
        raw_opening = self.memory_equivalence_opening
        bounded_opening = raw_opening.clamp(0.0, 1.0)
        opening = raw_opening + (bounded_opening - raw_opening).detach()
        return inherited + opening * (proposal - inherited)

    def memory_read_probability(
            self, features: torch.Tensor) -> torch.Tensor:
        """Return a generic read/no-read probability from memory statistics."""
        if self.memory_read_gate is None:
            raise RuntimeError("adaptive memory read is not enabled")
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("memory read features must have shape [batch, 4]")
        return torch.sigmoid(
            self.memory_read_gate(features)).squeeze(-1)

    def memory_replacement_scores(
            self, option_features: torch.Tensor) -> torch.Tensor:
        """Score skip/replace options from generic latent-memory statistics."""
        if self.memory_replacement_gate is None:
            raise RuntimeError("adaptive memory replacement is not enabled")
        if (
                option_features.ndim != 3
                or option_features.shape[-1]
                != self.adaptive_memory_replace_features):
            raise ValueError(
                "replacement features must have shape "
                f"[batch, options, {self.adaptive_memory_replace_features}]")
        # The leading slice is contiguous only when no extra feature is
        # present. Materializing it keeps the base-five scores bit-identical
        # across widths, so adding a zero-initialized statistic is exactly
        # behavior-preserving rather than merely close.
        scores = self.memory_replacement_gate(
            option_features[..., :5].contiguous()).squeeze(-1)
        if self.memory_replacement_extra_gate is not None:
            scores = scores + self.memory_replacement_extra_gate(
                option_features[..., 5:]).squeeze(-1)
        return scores

    def initial_state(
            self, batch_size: int, *, device: torch.device | str,
            dtype: torch.dtype = torch.float32) -> ControllerState:
        return ControllerState(
            torch.zeros(batch_size, self.width, device=device, dtype=dtype),
            torch.zeros(
                batch_size, self.workspace_slots, self.width,
                device=device, dtype=dtype),
        )

    def step(
            self, frame: torch.Tensor, state: ControllerState,
            previous_action: torch.Tensor, previous_reward: torch.Tensor,
            has_feedback: torch.Tensor,
            retrieved_memory: torch.Tensor | None = None,
            *, disable_workspace: bool = False) -> tuple[
                ControllerOutput, ControllerState]:
        event = self.vision(frame)
        if retrieved_memory is None:
            retrieved_memory = torch.zeros_like(event)
        query = torch.nn.functional.normalize(
            self.read_query(torch.cat([event, state.hidden], dim=-1)),
            dim=-1)
        slots = torch.nn.functional.normalize(
            state.workspace, dim=-1)
        scores = torch.einsum("bw,bsw->bs", query, slots)
        weights = torch.softmax(scores, dim=-1)
        read = torch.einsum("bs,bsw->bw", weights, state.workspace)
        if disable_workspace:
            read = torch.zeros_like(read)

        action_embedding = self.action_embedding(previous_action)
        feedback = self.feedback_encoder(torch.stack([
            previous_reward, has_feedback], dim=-1))
        controller_input = torch.cat([
            event, read, retrieved_memory, action_embedding, feedback], dim=-1)
        hidden = self.controller(controller_input, state.hidden)

        write_context = torch.cat([event, hidden, read], dim=-1)
        write_query = torch.nn.functional.normalize(
            self.write_query(write_context), dim=-1)
        write_scores = torch.einsum("bw,bsw->bs", write_query, slots)
        write_weights = torch.softmax(write_scores, dim=-1)
        gate = torch.sigmoid(self.write_gate(write_context))
        candidate = self.write_value(write_context)
        update = gate.unsqueeze(-1) * write_weights.unsqueeze(-1)
        workspace = (
            state.workspace * (1.0 - update)
            + candidate.unsqueeze(1) * update)
        if disable_workspace:
            workspace = torch.zeros_like(workspace)

        combined = torch.cat([hidden, read, event], dim=-1)
        intention = self.intention(combined)
        if self.relation_adapter is not None:
            relation_features = torch.cat([state.hidden, event], dim=-1)
            relation_residual = self.relation_adapter(relation_features)
            if self.relation_adapter_gate is not None:
                relation_residual = relation_residual * torch.sigmoid(
                    self.relation_adapter_gate(relation_features))
            intention = intention + relation_residual
        skill_adapter_openings = None
        skill_adapter_residual_norms = None
        if len(self.skill_adapters):
            # Successor slots read the same generic prior-state/query-event
            # pair as the legacy adapters: no task, context, or action label.
            slot_features = torch.cat([state.hidden, event], dim=-1)
            openings = []
            residual_norms = []
            prior_reads: list[torch.Tensor] = []
            # The legacy adapters hold what rungs two and three consolidated.
            # Their hidden layers are read on the same terms as a slot's: their
            # writes remain gated exactly as before.
            legacy_reads: list[torch.Tensor] = []
            if self.skill_adapter_legacy_read_from is not None:
                if self.relation_adapter is not None:
                    legacy_reads.append(
                        self.relation_adapter[1](
                            self.relation_adapter[0](slot_features)))
                if self.action_adapter is not None:
                    legacy_reads.append(
                        self.action_adapter[1](
                            self.action_adapter[0](slot_features)))
            for slot_index, (adapter, gate) in enumerate(zip(
                    self.skill_adapters, self.skill_adapter_gates)):
                # A slot sees the generic event pair plus what earlier slots
                # computed. The read is ungated on purpose: an earlier slot's
                # gate decides whether it speaks, not whether it can be
                # consulted. Without this, an exactly shut gate makes every
                # deeper ancestry produce bit-identical inputs here, and a new
                # slot has nothing to inherit.
                reads = []
                if (self.skill_adapter_reads_prior and prior_reads
                        and (self.skill_adapter_reads_prior_from is None
                             or slot_index
                             >= self.skill_adapter_reads_prior_from)):
                    reads += (
                        prior_reads[-self.skill_adapter_prior_read_limit:]
                        if self.skill_adapter_prior_read_limit
                        else prior_reads)
                if (self.skill_adapter_legacy_read_from is not None
                        and slot_index >= self.skill_adapter_legacy_read_from):
                    reads += legacy_reads
                if reads:
                    if self.skill_adapter_ablate_prior_read:
                        reads = [torch.zeros_like(r) for r in reads]
                    read_vector = torch.cat(reads, dim=-1)
                    projection = self.skill_adapter_read_projections[slot_index]
                    own_features = torch.cat(
                        [slot_features, projection(read_vector)], dim=-1)
                else:
                    own_features = slot_features
                score = gate(own_features)
                opening = (
                    # leaky_relu at slope zero is exactly relu, so a finished
                    # anneal restores exact-zero gating bit for bit.
                    torch.nn.functional.leaky_relu(
                        score, self.skill_adapter_gate_leak)
                    if self.skill_adapter_gate_mode == "relu"
                    else torch.sigmoid(score))
                hidden_read = adapter[1](adapter[0](own_features))
                residual = adapter[2](hidden_read) * opening
                if self.skill_adapter_reads_prior:
                    prior_reads.append(hidden_read)
                openings.append(opening)
                residual_norms.append(
                    residual.norm(dim=-1, keepdim=True))
                intention = intention + residual
            skill_adapter_openings = torch.cat(openings, dim=-1)
            skill_adapter_residual_norms = torch.cat(residual_norms, dim=-1)
        memory_context = torch.cat([hidden, read], dim=-1)
        logits = self.actuator(intention)
        if self.action_adapter is not None:
            adapter_features = torch.cat([state.hidden, event], dim=-1)
            adapter_residual = self.action_adapter(adapter_features)
            if self.action_adapter_gate is not None:
                adapter_residual = adapter_residual * torch.sigmoid(
                    self.action_adapter_gate(adapter_features))
            logits = logits + adapter_residual
        output = ControllerOutput(
            logits=logits,
            intention=intention,
            memory_key=self.memory_key(memory_context),
            memory_value=self.memory_value(memory_context),
            memory_write_strength=torch.sigmoid(
                self.memory_write(memory_context)).squeeze(-1),
            workspace_read=read,
            skill_adapter_openings=skill_adapter_openings,
            skill_adapter_residual_norms=skill_adapter_residual_norms,
        )
        return output, ControllerState(hidden, workspace)
