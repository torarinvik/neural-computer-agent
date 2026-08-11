"""End-to-end composition of the canonical runtime and keypress boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neural_computer import (
    AdaptiveOnlineEpisodicRelationReader,
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    CapabilityRetentionLedger,
    ContentAddressedMemory,
    ControllerFeedback,
    EpisodicContextEncoder,
    EpisodicContextOutput,
    EpisodicIntentAdapter,
    ExternalIntentionObservationReceipt,
    ExternalIntentionProposal,
    ExternalIntentionRepertoire,
    ExternalWorkingMemoryCell,
    KeypressDecoder,
    KeypressEncoder,
    OnlineEpisodicRelationReader,
    OpaqueViewRouteExtension,
    PersistentOpaqueContextRouteEvidence,
    PersistentOpaqueRouteEvidence,
    RetentionPolicyConfig,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier

ROUTE_STATE_SCHEMA = "neural-computer.brainworkshop-route-state.v1"
INTENTION_STATE_SCHEMA = "neural-computer.brainworkshop-intention-state.v1"


@dataclass(frozen=True)
class CanonicalRollout:
    """Auditable learner-visible trace from one verifier batch."""

    events: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    eligible: torch.Tensor
    propensities: torch.Tensor
    context: torch.Tensor
    episode_scores: torch.Tensor
    selected_slots: torch.Tensor

    @property
    def eligible_accuracy(self) -> torch.Tensor:
        count = self.eligible.sum(dim=1)
        correct = (self.rewards * self.eligible.to(self.rewards.dtype)).sum(dim=1)
        return correct / count.clamp_min(1)

    @property
    def replayed_examples(self) -> int:
        return 0


class RelationCapabilityExtension(nn.Module):
    """One append-only, replaceable external capability slot.

    The slot owns only memory-side state.  Its decoder is registered on the
    shared intention bus by :meth:`CanonicalBrainWorkshopAgent.add_relation_capability`;
    the controller and event frontend remain outside the slot.
    """

    def __init__(
        self,
        *,
        event_width: int,
        intention_width: int,
        action_width: int,
        memory_capacity: int,
        adaptive_reader: bool,
        decoder_name: str,
        seed: int,
        working_memory_cell: ExternalWorkingMemoryCell | None = None,
    ) -> None:
        super().__init__()
        if memory_capacity < 1:
            raise ValueError("extension memory capacity must be positive")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if working_memory_cell is None:
                reader_type = (
                    AdaptiveOnlineEpisodicRelationReader
                    if adaptive_reader
                    else OnlineEpisodicRelationReader
                )
                self.reader = reader_type(
                    event_width,
                    action_width,
                    memory_capacity=memory_capacity,
                    context_width=event_width,
                    hidden=max(16, event_width),
                )
            else:
                if (
                    working_memory_cell.event_width != event_width
                    or working_memory_cell.action_width != action_width
                    or working_memory_cell.memory_capacity != memory_capacity
                    or working_memory_cell.context_width != event_width
                ):
                    raise ValueError(
                        "working-memory cell dimensions do not match extension"
                    )
                self.reader = working_memory_cell
            self.intent_adapter = EpisodicIntentAdapter(
                event_width,
                intention_width,
                hidden=max(16, event_width),
            )
            self.route_score = OpaqueViewRouteExtension(
                event_width,
                hidden=max(16, event_width),
            )
            self.capability_key = nn.Parameter(torch.randn(event_width))
        self.memory_capacity = int(memory_capacity)
        self.decoder_name = decoder_name


class CanonicalBrainWorkshopAgent(nn.Module):
    """Production runtime with external episodic state and keypress I/O.

    The controller and output decoder are ordinary trainable modules for later
    experiments.  This runner itself performs no optimizer update: it is the
    canonical transport and accounting boundary used by future replay-free
    continual-learning trainers.
    """

    def __init__(
        self,
        *,
        symbol_count: int = 4,
        event_width: int = 32,
        intention_width: int = 8,
        feedback_width: int = 8,
        n_back: int = 4,
        memory_capacity: int = 8,
        retention_config: RetentionPolicyConfig | None = None,
        reader_kind: str = "context",
        seed: int = 0,
        intention_repertoire: ExternalIntentionRepertoire | None = None,
        working_memory_cell: ExternalWorkingMemoryCell | None = None,
    ) -> None:
        super().__init__()
        if n_back < 1:
            raise ValueError("n-back must be positive")
        if reader_kind not in {"context", "relation"}:
            raise ValueError("reader_kind must be context or relation")
        torch.manual_seed(seed)
        controller = AmodalCognitiveController(
            width=event_width,
            workspace_slots=max(2, n_back),
            intention_width=intention_width,
            feedback_width=feedback_width,
            event_window_capacity=max(4, n_back + 1),
            memory_top_k=1,
        )
        retention = CapabilityRetentionLedger(event_width, config=retention_config)
        self.keypress_encoder = KeypressEncoder(
            NBackVerifier.action_count,
            feedback_width,
        )
        self.episodic_context = EpisodicContextEncoder(
            event_width,
            NBackVerifier.action_count,
            hidden=max(16, event_width),
            context_width=event_width,
        )
        self.intent_adapter = EpisodicIntentAdapter(
            event_width,
            intention_width,
            hidden=max(16, event_width),
        )
        self.capability_key = nn.Parameter(torch.randn(event_width))
        self.runtime = AmodalControllerRuntime(
            controller,
            encoders={
                "stimulus": BrainWorkshopEventEncoder(symbol_count, event_width)
            },
            output_bus=AmodalOutputBus(
                {
                    "keypress": KeypressDecoder(
                        intention_width,
                        NBackVerifier.action_count,
                        hidden=max(8, intention_width),
                    )
                }
            ),
            memory=ContentAddressedMemory(
                event_width,
                capacity=memory_capacity,
                retention_ledger=retention,
            ),
        )
        self.n_back = int(n_back)
        self.reader_kind = reader_kind
        if intention_repertoire is not None and (
            intention_repertoire.width != intention_width
        ):
            raise ValueError("intention repertoire width does not match the agent")
        # This object is deliberately not an nn.Module child.  It is external,
        # caller-owned state that can grow, persist, and be replaced without
        # changing the controller checkpoint or its gradients.
        self.intention_repertoire = (
            intention_repertoire
            if intention_repertoire is not None
            else ExternalIntentionRepertoire(intention_width)
        )
        if working_memory_cell is not None:
            if reader_kind != "relation":
                raise ValueError(
                    "an external working-memory cell requires relation reader mode"
                )
            if (
                working_memory_cell.event_width != event_width
                or working_memory_cell.action_width != NBackVerifier.action_count
                or working_memory_cell.context_width != event_width
            ):
                raise ValueError(
                    "working-memory cell dimensions do not match the canonical agent"
                )
            self.relation_reader = working_memory_cell
        else:
            self.relation_reader = OnlineEpisodicRelationReader(
                event_width,
                NBackVerifier.action_count,
                memory_capacity=max(2, n_back + 1),
                context_width=event_width,
                hidden=max(16, event_width),
            )
        self.extensions = nn.ModuleList()
        self.route_evidence = PersistentOpaqueRouteEvidence()
        self.route_evidence.append_slot()
        self.context_route_evidence = PersistentOpaqueContextRouteEvidence(
            event_width
        )
        self.context_route_evidence.append_slot()

    @property
    def controller(self) -> AmodalCognitiveController:
        return self.runtime.controller

    @property
    def external_reader(self) -> nn.Module:
        return (
            self.episodic_context
            if self.reader_kind == "context"
            else self.relation_reader
        )

    @property
    def working_memory_cell(self) -> ExternalWorkingMemoryCell | None:
        """Return the versioned causal cell when one owns relation state."""

        if isinstance(self.relation_reader, ExternalWorkingMemoryCell):
            return self.relation_reader
        return None

    @property
    def keypress_decoder(self) -> KeypressDecoder:
        decoder = self.runtime.output_bus.decoders["keypress"]
        if not isinstance(decoder, KeypressDecoder):
            raise TypeError("keypress output bus has an incompatible decoder")
        return decoder

    @property
    def retention(self) -> CapabilityRetentionLedger:
        memory = self.runtime.memory
        if not isinstance(memory, ContentAddressedMemory):
            raise TypeError("canonical Brain Workshop runtime needs content memory")
        return memory.retention

    @property
    def capability_address(self) -> torch.Tensor:
        """Return the stable opaque address owned by this external capability."""

        return torch.nn.functional.normalize(self.capability_key.detach(), dim=0)

    def capability_address_for(self, slot: int) -> torch.Tensor:
        """Return an opaque address for slot zero or an appended extension."""

        if slot == 0:
            return self.capability_address
        if slot < 0 or slot > len(self.extensions):
            raise IndexError("capability slot is outside the append-only bank")
        return torch.nn.functional.normalize(
            self.extensions[slot - 1].capability_key.detach(), dim=0
        )

    def extension_decoder(self, slot: int) -> KeypressDecoder:
        """Return the output-bus decoder owned by an appended slot."""

        if slot < 1 or slot > len(self.extensions):
            raise IndexError("capability slot is outside the append-only bank")
        decoder = self.runtime.output_bus.decoders[
            self.extensions[slot - 1].decoder_name
        ]
        if not isinstance(decoder, KeypressDecoder):
            raise TypeError("extension decoder is not a keypress decoder")
        return decoder

    def _append_relation_capability(
        self,
        *,
        memory_capacity: int,
        seed: int,
        adaptive_reader: bool = False,
        working_memory_cell: ExternalWorkingMemoryCell | None = None,
    ) -> int:
        """Append one relation capability without changing the core width."""

        slot = len(self.extensions) + 1
        decoder_name = f"keypress_extension_{slot}"
        extension = RelationCapabilityExtension(
            event_width=self.controller.width,
            intention_width=self.controller.intention_width,
            action_width=NBackVerifier.action_count,
            memory_capacity=memory_capacity,
            adaptive_reader=adaptive_reader,
            decoder_name=decoder_name,
            seed=seed,
            working_memory_cell=working_memory_cell,
        )
        decoder = KeypressDecoder(
            self.controller.intention_width,
            NBackVerifier.action_count,
            hidden=max(8, self.controller.intention_width),
        )
        self.extensions.append(extension)
        self.runtime.register_decoder(decoder_name, decoder)
        self.route_evidence.append_slot()
        self.context_route_evidence.append_slot()
        return slot

    def add_relation_capability(self, *, n_back: int, seed: int) -> int:
        """Append a benchmark-sized relation capability.

        This compatibility path remains for the historical Brain Workshop
        ladder.  New growth experiments should use
        :meth:`add_adaptive_relation_capability`, which provisions only a
        bounded external event window and never receives an n-back value.
        """

        if n_back < 1:
            raise ValueError("extension n-back must be positive")
        return self._append_relation_capability(
            memory_capacity=max(2, n_back + 1),
            seed=seed,
        )

    def add_adaptive_relation_capability(
        self,
        *,
        memory_capacity: int,
        seed: int,
        working_memory_cell: ExternalWorkingMemoryCell | None = None,
    ) -> int:
        """Append a generic bounded-window capability without task metadata."""

        if memory_capacity < 1:
            raise ValueError("adaptive capability memory capacity must be positive")
        return self._append_relation_capability(
            memory_capacity=memory_capacity,
            seed=seed,
            adaptive_reader=True,
            working_memory_cell=working_memory_cell,
        )

    def expand_adaptive_relation_capability(
        self,
        slot: int,
        *,
        memory_capacity: int,
        reset_failed_reader: bool = False,
        reset_seed: int | None = None,
    ) -> None:
        """Grow one adaptive capability without changing shared state.

        When ``reset_failed_reader`` is true, the entire unmastered external
        slot is replaced at the larger capacity.  A failed reader can have
        already damaged its intention adapter or decoder, so resetting only
        the reader is not a valid capacity-growth transaction.  Mastered
        slots remain separate and are never reset by this operation.
        """

        if slot < 1 or slot > len(self.extensions):
            raise IndexError("adaptive capability slot is outside the bank")
        extension = self.extensions[slot - 1]
        if not isinstance(
            extension.reader,
            (AdaptiveOnlineEpisodicRelationReader, ExternalWorkingMemoryCell),
        ):
            raise TypeError("only adaptive relation capabilities can grow")
        if reset_failed_reader:
            replacement_seed = (
                int(torch.initial_seed()) if reset_seed is None else int(reset_seed)
            )
            working_memory_cell = None
            if isinstance(extension.reader, ExternalWorkingMemoryCell):
                with torch.random.fork_rng():
                    torch.manual_seed(replacement_seed)
                    working_memory_cell = ExternalWorkingMemoryCell(
                        self.controller.width,
                        NBackVerifier.action_count,
                        memory_capacity=memory_capacity,
                        context_width=self.controller.width,
                        hidden=extension.reader.hidden,
                    )
            replacement = RelationCapabilityExtension(
                event_width=self.controller.width,
                intention_width=self.controller.intention_width,
                action_width=NBackVerifier.action_count,
                memory_capacity=memory_capacity,
                adaptive_reader=True,
                decoder_name=extension.decoder_name,
                seed=replacement_seed,
                working_memory_cell=working_memory_cell,
            )
            self.extensions[slot - 1] = replacement
            with torch.random.fork_rng():
                torch.manual_seed(replacement_seed + 1)
                self.runtime.output_bus.decoders[extension.decoder_name] = (
                    KeypressDecoder(
                        self.controller.intention_width,
                        NBackVerifier.action_count,
                        hidden=max(8, self.controller.intention_width),
                    )
                )
            return
        if isinstance(extension.reader, ExternalWorkingMemoryCell):
            extension.reader = extension.reader.grow(memory_capacity)
        else:
            extension.reader = extension.reader.expand_capacity(
                memory_capacity,
                preserve_weights=True,
            )
        extension.memory_capacity = memory_capacity

    def replace_unprotected_adaptive_relation_capability(
        self,
        slot: int,
        *,
        memory_capacity: int,
        seed: int,
    ) -> dict[str, object]:
        """Evict one unprotected slot and install a fresh adaptive capability.

        The retention ledger is the safety gate.  Protected capabilities are
        rejected before any route evidence or neural slot state changes.  For
        an accepted replacement, both opaque route ledgers are cleared for
        the reused physical slot so an old cue cannot select the new
        capability by stale evidence.
        """

        if slot < 1 or slot > len(self.extensions):
            raise IndexError("adaptive capability slot is outside the bank")
        extension = self.extensions[slot - 1]
        if not isinstance(
            extension.reader,
            (AdaptiveOnlineEpisodicRelationReader, ExternalWorkingMemoryCell),
        ):
            raise TypeError("only adaptive relation capabilities can be replaced")
        if memory_capacity < 1:
            raise ValueError("replacement memory capacity must be positive")
        old_address = self.capability_address_for(slot)
        old_status = self.retention.status(old_address)
        route_status = self.route_evidence.status()
        context_protected = self.context_route_evidence.protected_slots()
        route_protected = route_status.protected[slot] or context_protected[slot]
        if old_status.protected or route_protected:
            raise ValueError("protected capability cannot be evicted")
        self.route_evidence.reset_slot(slot)
        self.context_route_evidence.reset_slot(slot)
        self.expand_adaptive_relation_capability(
            slot,
            memory_capacity=memory_capacity,
            reset_failed_reader=True,
            reset_seed=seed,
        )
        return {
            "slot": slot,
            "evicted_key_digest": old_status.key_digest,
            "evicted_protected": old_status.protected,
            "evicted_route_protected": route_protected,
            "replacement_key_digest": self.retention.status(
                self.capability_address_for(slot)
            ).key_digest,
        }

    def route_state_payload(self) -> dict[str, object]:
        """Return independently reloadable external route state."""

        return {
            "schema": ROUTE_STATE_SCHEMA,
            "slot_count": len(self.extensions) + 1,
            "route_evidence": self.route_evidence.payload(),
            "context_route_evidence": self.context_route_evidence.payload(),
        }

    def load_route_state_payload(self, payload: dict[str, object]) -> None:
        """Restore external route state without touching neural weights."""

        if payload.get("schema") != ROUTE_STATE_SCHEMA:
            raise ValueError("route-state schema is incompatible")
        slot_count = payload.get("slot_count")
        if slot_count != len(self.extensions) + 1:
            raise ValueError("route-state slot count does not match the agent")
        route_payload = payload.get("route_evidence")
        context_payload = payload.get("context_route_evidence")
        if not isinstance(route_payload, dict) or not isinstance(
            context_payload, dict
        ):
            raise TypeError("route-state ledgers must be dictionaries")
        route_evidence = PersistentOpaqueRouteEvidence.from_payload(route_payload)
        context_route_evidence = PersistentOpaqueContextRouteEvidence.from_payload(
            context_payload
        )
        if route_evidence.slot_count != slot_count:
            raise ValueError("route-state evidence has the wrong slot count")
        if context_route_evidence.slot_count != slot_count:
            raise ValueError("context route-state has the wrong slot count")
        if context_route_evidence.width != self.controller.width:
            raise ValueError("context route-state width does not match the agent")
        self.route_evidence = route_evidence
        self.context_route_evidence = context_route_evidence

    def intention_state_payload(self) -> dict[str, object]:
        """Return independently reloadable opaque intention-memory state."""

        return {
            "schema": INTENTION_STATE_SCHEMA,
            "repertoire": self.intention_repertoire.payload(),
        }

    def load_intention_state_payload(self, payload: dict[str, object]) -> None:
        """Restore intention memory without touching neural weights."""

        if payload.get("schema") != INTENTION_STATE_SCHEMA:
            raise ValueError("intention-state schema is incompatible")
        repertoire_payload = payload.get("repertoire")
        if not isinstance(repertoire_payload, dict):
            raise TypeError("intention-state repertoire must be a dictionary")
        repertoire = ExternalIntentionRepertoire.from_payload(repertoire_payload)
        if repertoire.width != self.controller.intention_width:
            raise ValueError("intention-state width does not match the agent")
        self.intention_repertoire = repertoire

    def observe_intention(
        self,
        intention: torch.Tensor,
        *,
        utility: torch.Tensor | float | None = None,
        propensity: torch.Tensor | float | None = None,
        timestamp: torch.Tensor | int | None = None,
        outcome_mask: torch.Tensor | bool | None = None,
    ) -> ExternalIntentionObservationReceipt:
        """Write opaque output experience to external memory only."""

        return self.intention_repertoire.observe(
            intention,
            utility=utility,
            propensity=propensity,
            timestamp=timestamp,
            outcome_mask=outcome_mask,
        )

    def propose_intentions(
        self,
        seed_intention: torch.Tensor | None = None,
        *,
        include_seed: bool = False,
        max_candidates: int | None = None,
    ) -> ExternalIntentionProposal:
        """Retrieve runtime-sized opaque candidates from external memory."""

        return self.intention_repertoire.propose(
            seed_intention,
            include_seed=include_seed,
            max_candidates=max_candidates,
        )

    def initial_state(self, batch_size: int, *, device: torch.device | str) -> object:
        return self.controller.initial_state(batch_size, device=device)

    def initial_feedback(
        self, batch_size: int, *, device: torch.device | str
    ) -> ControllerFeedback:
        return ControllerFeedback(
            action=torch.zeros(
                batch_size,
                self.controller.feedback_width,
                device=device,
            ),
            reward=torch.zeros(batch_size, device=device),
            propensity=torch.ones(batch_size, device=device),
            has_feedback=torch.zeros(batch_size, device=device),
        )

    def rollout(
        self,
        verifier: NBackVerifier,
        *,
        sample: bool = True,
        reset_history: bool = False,
        record_retention: bool = True,
        exploration_probability: float = 0.0,
        forced_slot: int | None = None,
        learned_route: bool = False,
        persistent_route: bool = False,
        context_route: bool = False,
        record_context_route: bool = False,
        record_intention_memory: bool = False,
    ) -> CanonicalRollout:
        """Run one online episode without replay or optimizer updates.

        Retention observations are opt-in at the call site because acquisition
        rollouts contain exploration and should not permanently lower the
        stable-prefix promotion gate.

        ``forced_slot`` is reserved for an external candidate-specific
        retention audit. The deployed learner leaves it unset and uses only
        outcome-driven routing.
        """

        if not 0.0 <= exploration_probability < 1.0:
            raise ValueError("slot exploration probability must lie in [0, 1)")
        if learned_route and len(self.extensions) == 0:
            raise ValueError("learned routing needs at least one appended slot")
        if not isinstance(record_intention_memory, bool):
            raise TypeError("intention-memory recording flag must be boolean")
        if sum((learned_route, persistent_route, context_route)) > 1:
            raise ValueError("route policies are mutually exclusive")

        verifier.reset()
        state = self.initial_state(verifier.batch_size, device=verifier.device)
        readers: list[nn.Module] = [self.external_reader]
        readers.extend(extension.reader for extension in self.extensions)
        if forced_slot is not None and not 0 <= forced_slot < len(readers):
            raise IndexError("forced retention-audit slot is outside the bank")
        reader_states = [
            reader.initial_state(verifier.batch_size, device=verifier.device)
            for reader in readers
        ]
        feedback = self.initial_feedback(
            verifier.batch_size,
            device=verifier.device,
        )
        previous_actions = torch.zeros(
            verifier.batch_size,
            NBackVerifier.action_count,
            device=verifier.device,
        )
        context = torch.zeros(
            verifier.batch_size,
            self.episodic_context.context_width,
            device=verifier.device,
        )
        selected_slot = torch.zeros(
            verifier.batch_size,
            dtype=torch.long,
            device=verifier.device,
        )
        route_order: torch.Tensor | None = None
        context_route_order: torch.Tensor | None = None
        route_context: torch.Tensor | None = None
        route_cursor = torch.zeros_like(selected_slot)
        if persistent_route:
            route_order = torch.tensor(
                self.route_evidence.preferred_order(slot_count=len(readers)),
                dtype=torch.long,
                device=verifier.device,
            )
            selected_slot.fill_(int(route_order[0]))
        if forced_slot is not None:
            selected_slot.fill_(forced_slot)
        event_trace: list[torch.Tensor] = []
        action_trace: list[torch.Tensor] = []
        reward_trace: list[torch.Tensor] = []
        eligible_trace: list[torch.Tensor] = []
        propensity_trace: list[torch.Tensor] = []
        selected_slot_trace: list[torch.Tensor] = []

        while not verifier.done:
            if reset_history and verifier.position:
                state = self.initial_state(
                    verifier.batch_size,
                    device=verifier.device,
                )
                reader_states = [
                    reader.initial_state(
                        verifier.batch_size,
                        device=verifier.device,
                    )
                    for reader in readers
                ]
                feedback = self.initial_feedback(
                    verifier.batch_size,
                    device=verifier.device,
                )
                previous_actions = torch.zeros_like(previous_actions)
                selected_slot = torch.full_like(
                    selected_slot,
                    0 if forced_slot is None else forced_slot,
                )
                route_cursor.zero_()
                if persistent_route and route_order is not None:
                    selected_slot.fill_(int(route_order[0]))
            route_exploration = torch.zeros(
                verifier.batch_size,
                dtype=torch.bool,
                device=verifier.device,
            )
            route_probability = torch.ones(
                verifier.batch_size,
                device=verifier.device,
            )
            route_target_probability = torch.zeros_like(route_probability)
            route_origin_slot = selected_slot.clone()
            route_target_slot = (route_origin_slot + 1).clamp_max(
                len(readers) - 1
            )
            if (
                forced_slot is None
                and not learned_route
                and exploration_probability
                and len(readers) > 1
            ):
                eligible_feedback = feedback.has_feedback > 0.0
                can_advance = selected_slot < len(readers) - 1
                route_exploration = eligible_feedback & can_advance & (
                    torch.rand(
                        verifier.batch_size,
                        device=verifier.device,
                    )
                    < exploration_probability
                )
                route_probability = torch.where(
                    eligible_feedback & can_advance,
                    torch.where(
                        route_exploration,
                        torch.full_like(
                            route_probability, exploration_probability
                        ),
                        torch.full_like(
                            route_probability, 1.0 - exploration_probability
                        ),
                    ),
                    route_probability,
                )
                route_target_probability = torch.where(
                    eligible_feedback & can_advance,
                    torch.full_like(
                        route_target_probability, exploration_probability
                    ),
                    route_target_probability,
                )
                selected_slot = torch.where(
                    route_exploration,
                    selected_slot + 1,
                    selected_slot,
                )
            collection = self.runtime.encode_streams(
                {"stimulus": verifier.observation()}
            )
            if context_route and context_route_order is None:
                route_context = collection.payload[:, 0].detach().clone()
                context_route_order = torch.tensor(
                    [
                        self.context_route_evidence.preferred_order(row)
                        for row in route_context
                    ],
                    dtype=torch.long,
                    device=verifier.device,
                )
                if forced_slot is None:
                    selected_slot = context_route_order[:, 0]
                route_origin_slot = selected_slot.clone()
                route_target_slot = (route_origin_slot + 1).clamp_max(
                    len(readers) - 1
                )
            contexts: list[torch.Tensor] = []
            for index, reader in enumerate(readers):
                context_output, reader_states[index] = reader.step(
                    event=collection.payload[:, 0],
                    action=previous_actions,
                    outcome=feedback.reward,
                    state=reader_states[index],
                )
                contexts.append(
                    context_output.context
                    if isinstance(context_output, EpisodicContextOutput)
                    else context_output
                )
            if learned_route and forced_slot is None and len(readers) > 1:
                eligible_feedback = feedback.has_feedback > 0.0
                can_route_to_extension = route_origin_slot < len(readers) - 1
                route_query = torch.stack(contexts, dim=1).gather(
                    1,
                    route_origin_slot[:, None, None].expand(
                        -1, 1, contexts[0].shape[-1]
                    ),
                ).squeeze(1)
                route_target_slot = torch.full_like(
                    route_origin_slot,
                    len(readers) - 1,
                )
                route_target_probability = torch.sigmoid(
                    self.extensions[-1].route_score(route_query)
                )
                route_active = eligible_feedback & can_route_to_extension
                if sample:
                    route_to_extension = torch.rand(
                        verifier.batch_size,
                        device=verifier.device,
                    ) < route_target_probability
                else:
                    route_to_extension = route_target_probability >= 0.5
                route_to_extension = route_to_extension & route_active
                route_probability = torch.where(
                    route_active,
                    torch.where(
                        route_to_extension,
                        route_target_probability,
                        1.0 - route_target_probability,
                    ),
                    route_probability,
                )
                route_target_probability = torch.where(
                    route_active,
                    route_target_probability,
                    torch.zeros_like(route_target_probability),
                )
                selected_slot = torch.where(
                    route_to_extension,
                    route_target_slot,
                    selected_slot,
                )
            output, state = self.runtime.step_events(collection, state, feedback)
            slot_logits: list[torch.Tensor] = []
            for index, slot_context in enumerate(contexts):
                adapter = (
                    self.intent_adapter
                    if index == 0
                    else self.extensions[index - 1].intent_adapter
                )
                adapted_intention = adapter(output.intention, slot_context)
                decoded = self.runtime.output_bus(adapted_intention)
                decoder_name = (
                    "keypress"
                    if index == 0
                    else self.extensions[index - 1].decoder_name
                )
                slot_logits.append(decoded[decoder_name])
            all_logits = torch.stack(slot_logits, dim=1)
            selected_logits = all_logits.gather(
                1,
                selected_slot[:, None, None].expand(
                    -1, 1, NBackVerifier.action_count
                ),
            ).squeeze(1)
            decision = self.keypress_decoder.decide_from_logits(
                selected_logits, sample=sample
            )
            action_probabilities = torch.softmax(all_logits, dim=-1)
            selected_probabilities = action_probabilities.gather(
                1,
                selected_slot[:, None, None].expand(
                    -1, 1, NBackVerifier.action_count
                ),
            ).squeeze(1)
            route_current_probabilities = action_probabilities.gather(
                1,
                route_origin_slot[:, None, None].expand(
                    -1, 1, NBackVerifier.action_count
                ),
            ).squeeze(1)
            route_next_probabilities = action_probabilities.gather(
                1,
                route_target_slot[:, None, None].expand(
                    -1, 1, NBackVerifier.action_count
                ),
            ).squeeze(1)
            route_mixture = (
                (1.0 - route_target_probability[:, None])
                * route_current_probabilities
                + route_target_probability[:, None] * route_next_probabilities
            )
            mixed_probabilities = torch.where(
                (route_probability < 1.0)[:, None],
                route_mixture,
                selected_probabilities,
            )
            propensity = mixed_probabilities.gather(
                1, decision.key_index[:, None]
            ).squeeze(1)
            context = torch.stack(contexts, dim=1).gather(
                1,
                selected_slot[:, None, None].expand(
                    -1, 1, contexts[0].shape[-1]
                ),
            ).squeeze(1)
            scored = verifier.score(decision.key_index)
            if record_intention_memory:
                self.observe_intention(
                    output.intention.payload.detach(),
                    utility=scored.reward.detach(),
                    propensity=propensity.detach(),
                    timestamp=verifier.position,
                    outcome_mask=scored.eligible.detach(),
                )
            event_trace.append(collection.payload[:, 0])
            action_trace.append(decision.key_index)
            reward_trace.append(scored.reward)
            eligible_trace.append(scored.eligible)
            propensity_trace.append(propensity)
            selected_slot_trace.append(selected_slot)
            feedback = ControllerFeedback(
                action=self.keypress_encoder(decision.key_index),
                reward=scored.reward,
                propensity=propensity,
                has_feedback=scored.eligible.to(scored.reward.dtype),
            )
            previous_actions = torch.nn.functional.one_hot(
                decision.key_index,
                num_classes=NBackVerifier.action_count,
            ).to(collection.payload.dtype)
            if forced_slot is None and len(readers) > 1:
                failed = scored.eligible & (scored.reward < 0.5)
                if context_route and context_route_order is not None:
                    can_advance = route_cursor < len(readers) - 1
                    route_cursor = torch.where(
                        failed & can_advance,
                        route_cursor + 1,
                        route_cursor,
                    )
                    selected_slot = context_route_order.gather(
                        1, route_cursor[:, None]
                    ).squeeze(1)
                elif persistent_route and route_order is not None:
                    can_advance = route_cursor < len(readers) - 1
                    route_cursor = torch.where(
                        failed & can_advance,
                        route_cursor + 1,
                        route_cursor,
                    )
                    selected_slot = route_order[route_cursor]
                else:
                    can_advance = selected_slot < len(readers) - 1
                    selected_slot = torch.where(
                        failed & can_advance,
                        selected_slot + 1,
                        selected_slot,
                    )

        events = torch.stack(event_trace, dim=1)
        actions = torch.stack(action_trace, dim=1)
        rewards = torch.stack(reward_trace, dim=1)
        eligible = torch.stack(eligible_trace, dim=1)
        propensities = torch.stack(propensity_trace, dim=1)
        selected_slots = torch.stack(selected_slot_trace, dim=1)
        episode_scores = (
            (rewards * eligible.to(rewards.dtype)).sum(dim=1)
            / eligible.sum(dim=1).clamp_min(1)
        )
        final_slots = selected_slots[:, -1]
        if record_retention:
            for slot in range(len(readers)):
                for score in episode_scores[final_slots == slot]:
                    self.retention.observe(
                        self.capability_address_for(slot), score
                    )
        if persistent_route:
            for slot in range(len(readers)):
                for score in episode_scores[final_slots == slot]:
                    self.route_evidence.observe(slot, score)
        if context_route and record_context_route and route_context is not None:
            context_rows: list[torch.Tensor] = []
            slot_rows: list[int] = []
            outcome_rows: list[torch.Tensor] = []
            for row in range(rewards.shape[0]):
                for slot in range(len(readers)):
                    attempted = (selected_slots[row] == slot) & eligible[row]
                    if bool(attempted.any()):
                        context_rows.append(route_context[row])
                        slot_rows.append(slot)
                        outcome_rows.append(rewards[row][attempted].mean())
            if context_rows:
                self.context_route_evidence.observe_batch(
                    torch.stack(context_rows),
                    torch.tensor(
                        slot_rows,
                        dtype=torch.long,
                        device=route_context.device,
                    ),
                    torch.stack(outcome_rows),
                )
        return CanonicalRollout(
            events=events,
            actions=actions,
            rewards=rewards,
            eligible=eligible,
            propensities=propensities,
            context=context,
            episode_scores=episode_scores,
            selected_slots=selected_slots,
        )
