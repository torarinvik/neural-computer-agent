"""Source-preserving rendered Brain Workshop on the production live tick."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neural_computer import (
    AmodalEventCollection,
    CognitiveTickRuntime,
    IntentEvent,
    KeypressDecoder,
    LiveActionProposal,
    LiveActionReceipt,
    LiveInputBatch,
    LiveOutcomeEvent,
    LiveTickResult,
    ResolvedLiveOutcome,
)

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopObservation,
    RenderedBrainWorkshopVerifier,
)

RENDERED_LIVE_SCHEMA = "neural-computer.brainworkshop-rendered-live.v1"


class RenderedBrainWorkshopLiveDevice:
    """Live RGB/audio frontend plus opaque keypress backend."""

    def __init__(
        self,
        verifier: RenderedBrainWorkshopVerifier,
        encoders: RenderedBrainWorkshopEncoders,
        *,
        reverse_event_order: bool = False,
        drop_streams: tuple[str, ...] = (),
        action_permutation: tuple[int, ...] | None = None,
        randomized_outcome_seed: int | None = None,
    ) -> None:
        self.verifier = verifier
        self.encoders = encoders
        self.batch_size = 1
        self.event_width = encoders.event_width
        self.reverse_event_order = bool(reverse_event_order)
        if any(stream not in verifier.config.streams for stream in drop_streams):
            raise ValueError("cannot drop a stream absent from the verifier")
        self.drop_streams = frozenset(drop_streams)
        if self.drop_streams == frozenset(verifier.config.streams):
            raise ValueError("at least one rendered stream must remain")
        if action_permutation is not None and sorted(action_permutation) != list(
            range(verifier.action_count)
        ):
            raise ValueError("action permutation must cover every opaque action")
        self.action_permutation = action_permutation
        self._outcome_generator = (
            None
            if randomized_outcome_seed is None
            else torch.Generator().manual_seed(randomized_outcome_seed)
        )
        self._observation_pending = True
        self._pending_events: AmodalEventCollection | None = None
        self._outcomes: list[LiveOutcomeEvent] = []

    @property
    def done(self) -> bool:
        return (
            self.verifier.done
            and not self._observation_pending
            and not self._outcomes
        )

    def poll(self, now: float) -> LiveInputBatch:
        outcomes = tuple(self._outcomes)
        self._outcomes.clear()
        if self._observation_pending and not self.verifier.done:
            if self._pending_events is None:
                observation = self.verifier.observation()
                observation = RenderedBrainWorkshopObservation(
                    vision=(
                        None
                        if "vision" in self.drop_streams
                        else observation.vision
                    ),
                    audio=(
                        None if "audio" in self.drop_streams else observation.audio
                    ),
                )
                with torch.no_grad():
                    self._pending_events = self.encoders.encode(
                        observation,
                        now=now,
                        reverse_order=self.reverse_event_order,
                    )
            events = self._pending_events
        else:
            events = AmodalEventCollection.empty(1, self.event_width)
        return LiveInputBatch(events, outcomes, now)

    def emit(self, action: torch.Tensor, receipt: LiveActionReceipt) -> None:
        if self._pending_events is None or self.verifier.done:
            raise RuntimeError("rendered Brain Workshop action has no stimulus")
        executed = action
        if self.action_permutation is not None:
            executed = torch.tensor(
                [self.action_permutation[int(action.item())]], dtype=action.dtype
            )
        scored = self.verifier.score(executed)
        reward = scored.reward
        if self._outcome_generator is not None:
            reward = torch.randint(
                0,
                2,
                reward.shape,
                generator=self._outcome_generator,
            ).to(reward.dtype)
        self._outcomes.append(
            LiveOutcomeEvent(
                receipt_id=receipt.receipt_id,
                reward=torch.where(
                    scored.eligible,
                    reward,
                    torch.zeros_like(scored.reward),
                ),
                present=scored.eligible,
                observed_at=receipt.emitted_at,
                confidence=torch.ones(1),
            )
        )
        self._pending_events = None
        self._observation_pending = not self.verifier.done


@dataclass(frozen=True)
class _SourceTemporalCredit:
    source_key: torch.Tensor
    current: torch.Tensor
    history: torch.Tensor
    history_present: torch.Tensor
    address_index: torch.Tensor | None = None
    address_propensity: torch.Tensor | None = None


@dataclass(frozen=True)
class _MultistreamCreditState:
    sources: tuple[_SourceTemporalCredit, ...]


class SourcePreservingTemporalMachine(nn.Module):
    """Generic per-source temporal memory with permutation-invariant composition.

    Every opaque source key owns a separate chronological history. One shared
    temporal relation reader processes all sources. A shared source-conditioned
    transform runs before summation, so raw event tensors and unconditioned
    modality residuals are never averaged together.
    """

    def __init__(
        self,
        event_width: int,
        *,
        source_key_width: int,
        max_history: int,
        max_sources: int,
        action_count: int,
        intention_width: int = 12,
        hidden: int = 32,
        learning_rate: float = 3e-3,
        sample: bool = True,
        action_delay_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        if min(
            event_width,
            source_key_width,
            max_history,
            max_sources,
            action_count,
            intention_width,
            hidden,
        ) < 1:
            raise ValueError("source-preserving machine dimensions must be positive")
        if learning_rate <= 0.0 or action_delay_seconds < 0.0:
            raise ValueError("source-preserving learning rate must be positive")
        if action_count < 2:
            raise ValueError("source-preserving learning needs at least two actions")
        self.event_width = int(event_width)
        self.source_key_width = int(source_key_width)
        self.max_history = int(max_history)
        self.max_sources = int(max_sources)
        self.action_count = int(action_count)
        self.intention_width = int(intention_width)
        self.output_key = "keypress"
        self.sample = bool(sample)
        self.action_delay_seconds = float(action_delay_seconds)
        self.learning_enabled = True
        self.reset_history_each_tick = False
        self.relative_address_logits = nn.Parameter(torch.zeros(max_history))
        self.relation = nn.Sequential(
            nn.Linear(event_width * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, intention_width),
        )
        self.source_conditioner = nn.Sequential(
            nn.Linear(intention_width + source_key_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, intention_width),
        )
        self.decoder = KeypressDecoder(intention_width, action_count, hidden=hidden)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self._histories: dict[tuple[float, ...], list[torch.Tensor]] = {}
        self._bound_source_identities: set[tuple[float, ...]] | None = None
        self.model_version = 0
        self.optimizer_updates = 0
        self.unique_outcome_bits = 0
        self.last_loss: float | None = None
        self._scheduled_proposal: tuple[float, LiveActionProposal] | None = None

    @staticmethod
    def _identity(source_key: torch.Tensor) -> tuple[float, ...]:
        if source_key.shape[0] != 1:
            raise ValueError("first rendered live rung is batch-one")
        return tuple(float(value) for value in source_key[0].detach().cpu())

    def reset_history(self) -> None:
        self._histories.clear()
        self._scheduled_proposal = None

    def bind_executable_sources(
        self, source_keys: tuple[torch.Tensor, ...] | None
    ) -> None:
        """Bind this program to a subset of bus sources.

        Extra simultaneous events remain on the collection. They do not
        resize the controller or enter this program's temporal history.
        """

        if source_keys is None:
            self._bound_source_identities = None
            return
        identities = {self._identity(key) for key in source_keys}
        if not identities:
            raise ValueError("executable source binding cannot be empty")
        if len(identities) > self.max_sources:
            raise ValueError("executable source binding exceeds fixed source capacity")
        self._bound_source_identities = identities

    def _history_tensors(
        self,
        source_key: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self._histories.get(self._identity(source_key), [])
        newest_first = list(reversed(rows[-self.max_history :]))
        history = torch.zeros(
            1,
            self.max_history,
            self.event_width,
            dtype=current.dtype,
            device=current.device,
        )
        present = torch.zeros(
            1,
            self.max_history,
            dtype=torch.bool,
            device=current.device,
        )
        if newest_first:
            history[:, : len(newest_first)] = torch.stack(newest_first, dim=1)
            present[:, : len(newest_first)] = True
        return history, present

    def _source_relation(self, credit: _SourceTemporalCredit) -> torch.Tensor:
        if credit.address_index is None:
            address = self.relative_address_logits.unsqueeze(0)
            address = address.masked_fill(
                ~credit.history_present,
                torch.finfo(address.dtype).min,
            )
            any_history = credit.history_present.any(dim=1)
            address = torch.where(
                any_history[:, None], address, torch.zeros_like(address)
            )
            weights = address.softmax(dim=1) * credit.history_present.to(
                address.dtype
            )
            retrieved = (weights.unsqueeze(-1) * credit.history).sum(dim=1)
        else:
            gather_index = credit.address_index[:, None, None].expand(
                -1, 1, self.event_width
            )
            retrieved = credit.history.gather(1, gather_index).squeeze(1)
        relation = self.relation(
            torch.cat(
                (
                    credit.current,
                    retrieved,
                    (credit.current - retrieved).square(),
                    credit.current * retrieved,
                ),
                dim=-1,
            )
        )
        return self.source_conditioner(torch.cat((relation, credit.source_key), dim=-1))

    def _prepare_credit(
        self, credit: _MultistreamCreditState
    ) -> _MultistreamCreditState:
        """Bind any executable program choices before action generation."""

        return credit

    def _forward_credit(
        self,
        credit: _MultistreamCreditState,
    ) -> tuple[IntentEvent, torch.Tensor]:
        if not credit.sources:
            raise ValueError("multistream credit needs at least one source")
        conditioned = torch.stack(
            [self._source_relation(source) for source in credit.sources],
            dim=1,
        )
        # Scaling stabilizes magnitude as runtime source count changes. The
        # source-conditioned nonlinear vectors remain separately computed.
        composite = conditioned.sum(dim=1) / len(credit.sources) ** 0.5
        intention = IntentEvent(composite)
        return intention, self.decoder(intention)

    def _learn(self, outcomes: tuple[ResolvedLiveOutcome, ...]) -> None:
        losses: list[torch.Tensor] = []
        for resolved in outcomes:
            present = resolved.event.present
            self.unique_outcome_bits += int(present.sum().item())
            if not bool(present.any()):
                continue
            credit = resolved.proposal.credit_state
            if not isinstance(credit, _MultistreamCreditState):
                raise TypeError("rendered proposal has incompatible credit state")
            _intention, logits = self._forward_credit(credit)
            log_probabilities = logits.log_softmax(dim=-1)
            selected_log_probability = log_probabilities.gather(
                1,
                resolved.receipt.action.to(torch.long).unsqueeze(-1),
            ).squeeze(-1)
            selected_mask = torch.nn.functional.one_hot(
                resolved.receipt.action.to(torch.long),
                num_classes=self.action_count,
            ).to(torch.bool)
            other_log_probability = torch.logsumexp(
                log_probabilities.masked_fill(selected_mask, -torch.inf),
                dim=-1,
            )
            reward = resolved.event.reward
            loss = -(
                reward * selected_log_probability
                + (1.0 - reward) * other_log_probability
            )
            losses.append(loss[present].mean())
        if not losses or not self.learning_enabled:
            return
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer_updates += 1
        self.model_version += 1
        self.last_loss = float(loss.detach())

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: tuple[ResolvedLiveOutcome, ...],
        *,
        now: float,
        elapsed: float,
    ) -> tuple[LiveActionProposal, ...]:
        del elapsed
        self._learn(outcomes)
        proposals: list[LiveActionProposal] = []
        if self._scheduled_proposal is not None:
            due_at, scheduled = self._scheduled_proposal
            if now >= due_at:
                proposals.append(scheduled)
                self._scheduled_proposal = None
        if self.reset_history_each_tick:
            self._histories.clear()
        if events.payload.shape[1] == 0:
            return tuple(proposals)
        if self._scheduled_proposal is not None:
            raise RuntimeError("new event arrived before delayed action emission")
        if events.payload.shape[0] != 1 or events.source_key is None:
            raise ValueError("rendered live events require opaque source keys")
        if not bool(events.present.all()):
            raise ValueError("rendered live sources cannot contain padded events")
        credits: list[_SourceTemporalCredit] = []
        identities: set[tuple[float, ...]] = set()
        for index in range(events.payload.shape[1]):
            source_key = events.source_key[:, index].detach()
            identity = self._identity(source_key)
            if identity in identities:
                raise ValueError("a live tick cannot contain a source twice")
            identities.add(identity)
            if (
                self._bound_source_identities is not None
                and identity not in self._bound_source_identities
            ):
                continue
            current = events.payload[:, index].detach()
            history, present = self._history_tensors(source_key, current)
            credits.append(
                _SourceTemporalCredit(
                    source_key=source_key.clone(),
                    current=current.clone(),
                    history=history.clone(),
                    history_present=present.clone(),
                )
            )
        if not credits:
            return tuple(proposals)
        live_identities = {self._identity(item.source_key) for item in credits}
        if self._bound_source_identities is None:
            if len(self._histories.keys() | live_identities) > self.max_sources:
                raise ValueError("live source count exceeds fixed source capacity")
        elif live_identities - self._bound_source_identities:
            raise RuntimeError("unbound source entered temporal execution")
        # Canonicalization is opaque and makes composition independent of the
        # device's event enumeration order.
        credits.sort(key=lambda item: self._identity(item.source_key))
        credit = self._prepare_credit(_MultistreamCreditState(tuple(credits)))
        with torch.no_grad():
            intention, logits = self._forward_credit(credit)
            probabilities = logits.softmax(dim=-1)
            action = (
                torch.multinomial(probabilities, 1).squeeze(-1)
                if self.sample
                else probabilities.argmax(dim=-1)
            )
            propensity = probabilities.gather(1, action[:, None]).squeeze(1)
        for source in credits:
            identity = self._identity(source.source_key)
            rows = self._histories.setdefault(identity, [])
            rows.append(source.current.clone())
            if len(rows) > self.max_history:
                del rows[0]
        proposal = LiveActionProposal(
            intention=IntentEvent(intention.payload.detach()),
            action=action,
            propensity=propensity,
            output_key=self.output_key,
            model_version=self.model_version,
            credit_state=credit,
        )
        if self.action_delay_seconds > 0.0:
            self._scheduled_proposal = (now + self.action_delay_seconds, proposal)
        else:
            proposals.append(proposal)
        return tuple(proposals)


class FrozenControllerProgramMachine(SourcePreservingTemporalMachine):
    """Frozen executor/controller boundary plus one mutable neural program file.

    The generic controller in this live rung owns causal ticks, per-source
    history, receipt binding, and execution of the external file. It has no
    task-updated parameters. Temporal addressing, relation computation, source
    conditioning, and action intention weights are capability-file contents;
    Brain Workshop feedback may update those file tensors and their optimizer,
    but never the controller/executor contract.

    A production meta-trained controller can replace this parameter-free
    executor behind the same boundary. This class does not claim that such
    controller pretraining has already happened.
    """

    learning_target = "external_program_file"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.program_file_updates = 0
        self._frozen_controller_digest = self.controller_digest()

    def controller_digest(self) -> str:
        """Digest the immutable executor ABI, excluding external file tensors."""

        description = (
            "neural-computer.frozen-live-program-executor.v1",
            self.event_width,
            self.source_key_width,
            self.max_history,
            self.max_sources,
            self.action_count,
            self.intention_width,
            self.output_key,
        )
        return hashlib.sha256(repr(description).encode()).hexdigest()

    def program_digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def assert_controller_frozen(self) -> None:
        if self.controller_digest() != self._frozen_controller_digest:
            raise RuntimeError("frozen program executor changed")

    def _learn(self, outcomes: tuple[ResolvedLiveOutcome, ...]) -> None:
        self.assert_controller_frozen()
        super()._learn(outcomes)
        self.program_file_updates += self.optimizer_updates
        # ``optimizer_updates`` is reserved for the frozen controller. The
        # optimizer belongs to the independently persisted external file.
        self.optimizer_updates = 0
        self.assert_controller_frozen()

    def external_program_payload(self) -> dict[str, object]:
        self.assert_controller_frozen()
        return {
            "learning_target": self.learning_target,
            "controller_digest": self._frozen_controller_digest,
            "program_file_updates": self.program_file_updates,
            "program_digest": self.program_digest(),
            "state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_external_program_payload(self, payload: dict[str, object]) -> None:
        if payload.get("learning_target") != self.learning_target:
            raise ValueError("checkpoint is not an external program file")
        if payload.get("controller_digest") != self._frozen_controller_digest:
            raise ValueError("external program file targets another controller")
        updates = payload.get("program_file_updates")
        state = payload.get("state")
        optimizer_state = payload.get("optimizer_state")
        if (
            not isinstance(updates, int)
            or updates < 0
            or not isinstance(state, dict)
            or not isinstance(optimizer_state, dict)
        ):
            raise ValueError("external program checkpoint is malformed")
        self.load_state_dict(state)
        self.optimizer.load_state_dict(optimizer_state)
        self.program_file_updates = updates
        if payload.get("program_digest") != self.program_digest():
            raise ValueError("external program checkpoint digest mismatch")
        self.assert_controller_frozen()


class PretrainedControllerProgramMachine(SourcePreservingTemporalMachine):
    """Frozen learned temporal executor plus one mutable address program.

    Controller relation, source-conditioning, and intention-decoding weights
    must come from prior verified experience. The inherited program prior is
    immutable controller-artifact content used only by an explicit transfer
    control. A genuinely fresh task file starts uniformly and may update only
    its relative temporal-address logits.
    """

    learning_target = "external_temporal_address_program"
    _PROGRAM_PARAMETER = "relative_address_logits"

    def __init__(
        self,
        *args,
        controller_state: dict[str, torch.Tensor],
        program_prior: torch.Tensor,
        initialize_program_from_prior: bool = True,
        **kwargs,
    ) -> None:
        learning_rate = float(kwargs.get("learning_rate", 3e-3))
        super().__init__(*args, **kwargs)
        named = dict(self.named_parameters())
        controller_names = set(named) - {self._PROGRAM_PARAMETER}
        if set(controller_state) != controller_names:
            raise ValueError("controller artifact parameter names do not match")
        if program_prior.shape != self.relative_address_logits.shape:
            raise ValueError("controller program prior shape does not match")
        with torch.no_grad():
            for name in sorted(controller_names):
                value = controller_state[name]
                if value.shape != named[name].shape:
                    raise ValueError(
                        f"controller artifact shape does not match for {name}"
                    )
                named[name].copy_(value)
            self.relative_address_logits.copy_(
                program_prior
                if initialize_program_from_prior
                else torch.zeros_like(program_prior)
            )
        self.register_buffer(
            "inherited_program_prior", program_prior.detach().clone()
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == self._PROGRAM_PARAMETER)
        self.optimizer = torch.optim.Adam(
            (self.relative_address_logits,), lr=learning_rate
        )
        self.program_file_updates = 0
        self._frozen_controller_digest = self.controller_digest()

    @staticmethod
    def _update_tensor_digest(
        digest: Any, name: str, value: torch.Tensor
    ) -> None:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())

    def controller_digest(self) -> str:
        digest = hashlib.sha256()
        description = (
            "neural-computer.pretrained-temporal-controller.v2",
            self.event_width,
            self.source_key_width,
            self.max_history,
            self.max_sources,
            self.action_count,
            self.intention_width,
            self.output_key,
        )
        digest.update(repr(description).encode())
        for name, value in sorted(self.named_parameters()):
            if name != self._PROGRAM_PARAMETER:
                self._update_tensor_digest(digest, name, value)
        self._update_tensor_digest(
            digest, "inherited_program_prior", self.inherited_program_prior
        )
        return digest.hexdigest()

    def admitted_program_artifact(self):
        """Snapshot the learned address file for verifier-gated admission.

        The optimizer state remains provisional training state.  The durable
        executable contains only the learned address instruction tensor and
        its independently versioned runtime interfaces.
        """

        from neural_computer.program import ExternalProgramArtifact
        from neural_computer.temporal_program import (
            TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

        self.assert_controller_frozen()
        return ExternalProgramArtifact(
            codes=self.relative_address_logits.detach().cpu().unsqueeze(0),
            interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

    def load_admitted_program_artifact(
        self,
        artifact,
        *,
        controller_digest: str,
    ) -> None:
        """Activate a retrieved file for deterministic frozen execution."""

        from neural_computer.program import ExternalProgramArtifact
        from neural_computer.temporal_program import (
            TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError("admitted temporal program must be an external artifact")
        if controller_digest != self._frozen_controller_digest:
            raise ValueError("admitted temporal program bank targets another controller")
        artifact.validate_for(
            instruction_width=self.max_history,
            interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )
        if artifact.codes.shape != (1, self.max_history):
            raise ValueError("admitted temporal address program has the wrong shape")
        with torch.no_grad():
            self.relative_address_logits.copy_(
                artifact.codes[0].to(self.relative_address_logits)
            )
        self.learning_enabled = False
        self.sample = False
        self.assert_controller_frozen()

    def program_digest(self) -> str:
        digest = hashlib.sha256()
        self._update_tensor_digest(
            digest, self._PROGRAM_PARAMETER, self.relative_address_logits
        )
        return digest.hexdigest()

    def assert_controller_frozen(self) -> None:
        if self.controller_digest() != self._frozen_controller_digest:
            raise RuntimeError("pretrained controller artifact changed")

    def _prepare_credit(
        self, credit: _MultistreamCreditState
    ) -> _MultistreamCreditState:
        """Execute one logged categorical temporal address per source."""

        prepared: list[_SourceTemporalCredit] = []
        with torch.no_grad():
            for source in credit.sources:
                masked = self.relative_address_logits.unsqueeze(0).masked_fill(
                    ~source.history_present,
                    -torch.inf,
                )
                any_history = source.history_present.any(dim=1)
                safe = torch.where(any_history[:, None], masked, torch.zeros_like(masked))
                probabilities = safe.softmax(dim=1)
                address = (
                    torch.multinomial(probabilities, 1).squeeze(1)
                    if self.sample
                    else probabilities.argmax(dim=1)
                )
                propensity = probabilities.gather(1, address[:, None]).squeeze(1)
                prepared.append(
                    _SourceTemporalCredit(
                        source_key=source.source_key,
                        current=source.current,
                        history=source.history,
                        history_present=source.history_present,
                        address_index=address,
                        address_propensity=propensity,
                    )
                )
        return _MultistreamCreditState(tuple(prepared))

    def _learn(self, outcomes: tuple[ResolvedLiveOutcome, ...]) -> None:
        self.assert_controller_frozen()
        losses: list[torch.Tensor] = []
        for resolved in outcomes:
            present = resolved.event.present
            self.unique_outcome_bits += int(present.sum().item())
            if not bool(present.any()):
                continue
            credit = resolved.proposal.credit_state
            if not isinstance(credit, _MultistreamCreditState):
                raise TypeError("rendered proposal has incompatible credit state")
            reward = resolved.event.reward
            informed_losses: list[torch.Tensor] = []
            for source in credit.sources:
                if source.address_index is None or source.address_propensity is None:
                    raise TypeError("pretrained proposal lacks temporal-address credit")
                masked = self.relative_address_logits.unsqueeze(0).masked_fill(
                    ~source.history_present,
                    -torch.inf,
                )
                any_history = source.history_present.any(dim=1)
                safe = torch.where(any_history[:, None], masked, torch.zeros_like(masked))
                log_probabilities = safe.log_softmax(dim=1)
                selected = log_probabilities.gather(
                    1, source.address_index[:, None]
                ).squeeze(1)
                selected_mask = torch.nn.functional.one_hot(
                    source.address_index,
                    num_classes=self.max_history,
                ).to(torch.bool)
                other = torch.logsumexp(
                    log_probabilities.masked_fill(selected_mask, -torch.inf),
                    dim=1,
                )
                informed = source.history_present.sum(dim=1) > 1
                if bool((present & informed).any()):
                    categorical = -(reward * selected + (1.0 - reward) * other)
                    informed_losses.append(categorical[present & informed].mean())
            losses.append(
                torch.stack(informed_losses).mean()
                if informed_losses
                else self.relative_address_logits.sum() * 0.0
            )
        if not losses or not self.learning_enabled:
            return
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((self.relative_address_logits,), max_norm=1.0)
        self.optimizer.step()
        self.program_file_updates += 1
        self.model_version += 1
        self.last_loss = float(loss.detach())
        self.assert_controller_frozen()

    def external_program_payload(self) -> dict[str, object]:
        self.assert_controller_frozen()
        return {
            "learning_target": self.learning_target,
            "controller_digest": self._frozen_controller_digest,
            "program_file_updates": self.program_file_updates,
            "program_digest": self.program_digest(),
            "relative_address_logits": self.relative_address_logits.detach().clone(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_external_program_payload(self, payload: dict[str, object]) -> None:
        if payload.get("learning_target") != self.learning_target:
            raise ValueError("checkpoint is not a temporal-address program file")
        if payload.get("controller_digest") != self._frozen_controller_digest:
            raise ValueError("program file targets another pretrained controller")
        updates = payload.get("program_file_updates")
        logits = payload.get("relative_address_logits")
        optimizer_state = payload.get("optimizer_state")
        if (
            not isinstance(updates, int)
            or updates < 0
            or not isinstance(logits, torch.Tensor)
            or logits.shape != self.relative_address_logits.shape
            or not isinstance(optimizer_state, dict)
        ):
            raise ValueError("temporal-address program checkpoint is malformed")
        with torch.no_grad():
            self.relative_address_logits.copy_(logits)
        self.optimizer.load_state_dict(optimizer_state)
        self.program_file_updates = updates
        if payload.get("program_digest") != self.program_digest():
            raise ValueError("temporal-address program digest mismatch")
        self.assert_controller_frozen()


class RecursiveTemporalProgramMachine(PretrainedControllerProgramMachine):
    """Execute a reusable relative-step primitive recursively.

    One program row is one opaque relative-history step. Repeating the row
    composes the same learned operation: a primitive concentrated on offset
    one retrieves one-back at depth one and two-back at depth two. The frozen
    relation/controller weights are identical to the legacy executor; only the
    independently versioned external interpreter changes.
    """

    learning_target = "external_recursive_temporal_program"

    def __init__(self, *args, **kwargs) -> None:
        self.composition_depth = 1
        super().__init__(*args, **kwargs)

    def legacy_controller_digest(self) -> str:
        return PretrainedControllerProgramMachine.controller_digest(self)

    def controller_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.legacy_controller_digest().encode())
        digest.update(b"neural-computer.recursive-relative-history-executor.v1")
        return digest.hexdigest()

    def program_digest(self) -> str:
        digest = hashlib.sha256()
        self._update_tensor_digest(
            digest, self._PROGRAM_PARAMETER, self.relative_address_logits
        )
        digest.update(str(self.composition_depth).encode())
        digest.update(b"neural-computer.relative-history-compose.v1")
        return digest.hexdigest()

    def admitted_program_artifact(self):
        from neural_computer.program import ExternalProgramArtifact
        from neural_computer.temporal_program import (
            RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
            RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
            TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

        self.assert_controller_frozen()
        return ExternalProgramArtifact(
            codes=self.relative_address_logits.detach()
            .cpu()
            .unsqueeze(0)
            .repeat(self.composition_depth, 1),
            interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
            execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

    def load_legacy_primitive_artifact(
        self,
        artifact,
        *,
        controller_digest: str,
    ) -> None:
        """Behavior-preservingly lift one verified legacy row to depth one."""

        from neural_computer.program import ExternalProgramArtifact
        from neural_computer.temporal_program import (
            TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError("legacy temporal primitive must be an external artifact")
        if controller_digest != self.legacy_controller_digest():
            raise ValueError("legacy primitive targets another frozen controller")
        artifact.validate_for(
            instruction_width=self.max_history,
            interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
            execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )
        if artifact.program_length != 1:
            raise ValueError("legacy temporal primitive must contain one row")
        with torch.no_grad():
            self.relative_address_logits.copy_(
                artifact.codes[0].to(self.relative_address_logits)
            )
        self.composition_depth = 1
        self.learning_enabled = False
        self.sample = False
        self.assert_controller_frozen()

    def load_recursive_program_artifact(
        self,
        artifact,
        *,
        controller_digest: str,
    ) -> None:
        from neural_computer.program import ExternalProgramArtifact
        from neural_computer.temporal_program import (
            RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
            RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
            TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
            pad_recursive_temporal_program,
        )

        if not isinstance(artifact, ExternalProgramArtifact):
            raise TypeError("recursive temporal program must be an external artifact")
        if controller_digest != self._frozen_controller_digest:
            raise ValueError("recursive program targets another frozen controller")

        if artifact.instruction_width < self.max_history:
            artifact = pad_recursive_temporal_program(artifact, self.max_history)
        artifact.validate_for(
            instruction_width=self.max_history,
            interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
            execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )
        if artifact.program_length > self.max_history:
            raise ValueError("recursive temporal program exceeds history capacity")
        reference = artifact.codes[0].expand_as(artifact.codes)
        if not torch.equal(artifact.codes, reference):
            raise ValueError("recursive program rows must reuse one primitive")
        with torch.no_grad():
            self.relative_address_logits.copy_(
                artifact.codes[0].to(self.relative_address_logits)
            )
        self.composition_depth = artifact.program_length
        self.learning_enabled = False
        self.sample = False
        self.assert_controller_frozen()

    def _recursive_probabilities(
        self, source: _SourceTemporalCredit
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primitive = self.relative_address_logits.softmax(dim=0)
        zero = primitive.new_zeros(())
        distribution = torch.stack((primitive.new_ones(()), *([zero] * self.max_history)))
        for _step in range(self.composition_depth):
            values = [zero]
            for total in range(1, self.max_history + 1):
                terms = [
                    distribution[total - offset] * primitive[offset - 1]
                    for offset in range(1, total + 1)
                ]
                values.append(torch.stack(terms).sum())
            distribution = torch.stack(values)
        effective = distribution[1:].unsqueeze(0).expand(
            source.history_present.shape[0], -1
        )
        valid = source.history_present & (effective > 0.0)
        masked = effective * valid.to(effective.dtype)
        total = masked.sum(dim=1, keepdim=True)
        probabilities = torch.where(
            total > 0.0,
            masked / total.clamp_min(torch.finfo(masked.dtype).tiny),
            torch.full_like(masked, 1.0 / self.max_history),
        )
        return probabilities, valid

    def _prepare_credit(
        self, credit: _MultistreamCreditState
    ) -> _MultistreamCreditState:
        prepared: list[_SourceTemporalCredit] = []
        with torch.no_grad():
            for source in credit.sources:
                probabilities, _valid = self._recursive_probabilities(source)
                address = (
                    torch.multinomial(probabilities, 1).squeeze(1)
                    if self.sample
                    else probabilities.argmax(dim=1)
                )
                propensity = probabilities.gather(1, address[:, None]).squeeze(1)
                prepared.append(
                    _SourceTemporalCredit(
                        source_key=source.source_key,
                        current=source.current,
                        history=source.history,
                        history_present=source.history_present,
                        address_index=address,
                        address_propensity=propensity,
                    )
                )
        return _MultistreamCreditState(tuple(prepared))

    def _learn(self, outcomes: tuple[ResolvedLiveOutcome, ...]) -> None:
        self.assert_controller_frozen()
        losses: list[torch.Tensor] = []
        for resolved in outcomes:
            present = resolved.event.present
            self.unique_outcome_bits += int(present.sum().item())
            if not bool(present.any()):
                continue
            credit = resolved.proposal.credit_state
            if not isinstance(credit, _MultistreamCreditState):
                raise TypeError("recursive proposal has incompatible credit state")
            reward = resolved.event.reward
            informed_losses: list[torch.Tensor] = []
            for source in credit.sources:
                if source.address_index is None:
                    raise TypeError("recursive proposal lacks temporal-address credit")
                probabilities, valid = self._recursive_probabilities(source)
                log_probabilities = probabilities.clamp_min(
                    torch.finfo(probabilities.dtype).tiny
                ).log()
                selected = log_probabilities.gather(
                    1, source.address_index[:, None]
                ).squeeze(1)
                selected_mask = torch.nn.functional.one_hot(
                    source.address_index,
                    num_classes=self.max_history,
                ).to(torch.bool)
                other = torch.logsumexp(
                    log_probabilities.masked_fill(selected_mask | ~valid, -torch.inf),
                    dim=1,
                )
                informed = valid.sum(dim=1) > 1
                if bool((present & informed).any()):
                    categorical = -(reward * selected + (1.0 - reward) * other)
                    informed_losses.append(categorical[present & informed].mean())
            losses.append(
                torch.stack(informed_losses).mean()
                if informed_losses
                else self.relative_address_logits.sum() * 0.0
            )
        if not losses or not self.learning_enabled:
            return
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((self.relative_address_logits,), max_norm=1.0)
        self.optimizer.step()
        self.program_file_updates += 1
        self.model_version += 1
        self.last_loss = float(loss.detach())
        self.assert_controller_frozen()

    def external_program_payload(self) -> dict[str, object]:
        self.assert_controller_frozen()
        return {
            "learning_target": self.learning_target,
            "controller_digest": self._frozen_controller_digest,
            "program_file_updates": self.program_file_updates,
            "program_digest": self.program_digest(),
            "composition_depth": self.composition_depth,
            "relative_address_logits": self.relative_address_logits.detach().clone(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_external_program_payload(self, payload: dict[str, object]) -> None:
        if payload.get("learning_target") != self.learning_target:
            raise ValueError("checkpoint is not a recursive temporal program")
        if payload.get("controller_digest") != self._frozen_controller_digest:
            raise ValueError("recursive program targets another controller")
        updates = payload.get("program_file_updates")
        depth = payload.get("composition_depth")
        logits = payload.get("relative_address_logits")
        optimizer_state = payload.get("optimizer_state")
        if (
            not isinstance(updates, int)
            or updates < 0
            or not isinstance(depth, int)
            or not 1 <= depth <= self.max_history
            or not isinstance(logits, torch.Tensor)
            or logits.shape != self.relative_address_logits.shape
            or not isinstance(optimizer_state, dict)
        ):
            raise ValueError("recursive temporal checkpoint is malformed")
        with torch.no_grad():
            self.relative_address_logits.copy_(logits)
        self.optimizer.load_state_dict(optimizer_state)
        self.program_file_updates = updates
        self.composition_depth = depth
        if payload.get("program_digest") != self.program_digest():
            raise ValueError("recursive temporal checkpoint digest mismatch")
        self.assert_controller_frozen()


@dataclass(frozen=True)
class RenderedLiveLifetime:
    actions: torch.Tensor
    rewards: torch.Tensor
    outcome_present: torch.Tensor
    eligible_accuracy: float
    input_events: int
    unique_verifier_bits: int
    optimizer_updates: int
    program_file_updates: int
    replayed_examples: int
    ticks: int
    deadline_misses: int
    total_seconds_p50: float
    total_seconds_p99: float
    schema: str = RENDERED_LIVE_SCHEMA


def run_rendered_live_lifetime(
    machine: SourcePreservingTemporalMachine,
    encoders: RenderedBrainWorkshopEncoders,
    config: RenderedBrainWorkshopConfig,
    *,
    seed: int,
    learn: bool = True,
    sample: bool = True,
    tick_seconds: float = 0.01,
    reverse_event_order: bool = False,
    drop_streams: tuple[str, ...] = (),
    reset_history_each_tick: bool = False,
    action_permutation: tuple[int, ...] | None = None,
    randomized_outcome_seed: int | None = None,
    max_tick_seconds: float | None = None,
) -> RenderedLiveLifetime:
    """Run one never-replayed rendered lifetime on the live tick."""

    config.validate()
    if machine.action_count != config.action_count:
        raise ValueError("machine action count does not match rendered protocol")
    verifier = RenderedBrainWorkshopVerifier(config, seed=seed)
    device = RenderedBrainWorkshopLiveDevice(
        verifier,
        encoders,
        reverse_event_order=reverse_event_order,
        drop_streams=drop_streams,
        action_permutation=action_permutation,
        randomized_outcome_seed=randomized_outcome_seed,
    )
    machine.reset_history()
    machine.learning_enabled = learn
    machine.sample = sample
    machine.reset_history_each_tick = bool(reset_history_each_tick)
    bind = getattr(machine, "bind_executable_sources", None)
    remaining = [stream for stream in config.streams if stream not in drop_streams]
    if bind is not None and remaining and machine.max_sources == 1:
        bind((encoders.source_keys[remaining[0]].detach().reshape(1, -1),))
    runtime = CognitiveTickRuntime(
        device,
        machine,
        {"keypress": device},
        max_tick_seconds=max_tick_seconds,
    )
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    present: list[torch.Tensor] = []
    results: list[LiveTickResult] = []
    start_updates = machine.optimizer_updates
    start_program_updates = getattr(machine, "program_file_updates", 0)
    now = 0.0
    while not device.done or runtime.pending_receipts:
        result = runtime.tick(now)
        results.append(result)
        actions.extend(receipt.action for receipt in result.emitted_receipts)
        for resolved in result.resolved_outcomes:
            rewards.append(resolved.event.reward)
            present.append(resolved.event.present)
        if len(results) > config.steps + 2:
            raise RuntimeError("rendered live session failed to drain")
        now += tick_seconds
    action_tensor = torch.stack(actions, dim=1)
    reward_tensor = torch.stack(rewards, dim=1)
    present_tensor = torch.stack(present, dim=1)
    denominator = present_tensor.sum().clamp_min(1)
    accuracy = float(
        (reward_tensor * present_tensor.to(reward_tensor.dtype)).sum() / denominator
    )
    total_seconds = sorted(result.total_seconds for result in results)

    def percentile(fraction: float) -> float:
        index = min(len(total_seconds) - 1, int(fraction * len(total_seconds)))
        return total_seconds[index]

    return RenderedLiveLifetime(
        actions=action_tensor,
        rewards=reward_tensor,
        outcome_present=present_tensor,
        eligible_accuracy=accuracy,
        input_events=sum(result.input_event_count for result in results),
        unique_verifier_bits=int(present_tensor.sum().item()),
        optimizer_updates=machine.optimizer_updates - start_updates,
        program_file_updates=(
            getattr(machine, "program_file_updates", 0) - start_program_updates
        ),
        replayed_examples=0,
        ticks=len(results),
        deadline_misses=sum(int(result.deadline_missed) for result in results),
        total_seconds_p50=percentile(0.50),
        total_seconds_p99=percentile(0.99),
    )
