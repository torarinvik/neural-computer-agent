"""Opaque multi-stream routing over one shared factual memory bank."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .world_model import (
    ExternalOnlineTransitionContextResult,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionModelBank,
    ExternalTransitionModelCandidateReceipt,
    ExternalTransitionModelConsolidationReceipt,
    ExternalTransitionObservation,
)

EXTERNAL_MULTI_STREAM_TRANSITION_ROUTER_SCHEMA = (
    "neural-computer.external-multi-stream-transition-router.v1"
)


def _digest_value(digest: hashlib._Hash, value: object) -> None:
    """Hash nested payloads without relying on tensor ``repr`` formatting."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=str):
            _digest_value(digest, str(key))
            _digest_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        digest.update(repr(len(value)).encode("utf-8"))
        for item in value:
            _digest_value(digest, item)
        return
    if value is None:
        digest.update(b"none")
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def _state_digest(
    shared: Mapping[str, object], streams: list[dict[str, object]]
) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, shared)
    _digest_value(digest, streams)
    return digest.hexdigest()


def _validate_stream_key(stream_key: torch.Tensor, *, width: int) -> torch.Tensor:
    if not isinstance(stream_key, torch.Tensor):
        raise TypeError("multi-stream key must be a tensor")
    if stream_key.ndim != 1 or stream_key.shape[0] != width:
        raise ValueError("multi-stream key has the wrong shape")
    if not bool(torch.isfinite(stream_key).all()):
        raise ValueError("multi-stream key must be finite")
    if float(torch.linalg.vector_norm(stream_key)) <= 1e-12:
        raise ValueError("multi-stream key must be non-zero")
    return torch.nn.functional.normalize(
        stream_key.detach().to(device="cpu", dtype=torch.float32),
        dim=0,
    )


@dataclass(frozen=True)
class ExternalMultiStreamTransitionContextResult:
    """A stream-attributed result from the shared transition router."""

    stream_key: torch.Tensor
    result: ExternalOnlineTransitionContextResult
    schema: str = EXTERNAL_MULTI_STREAM_TRANSITION_ROUTER_SCHEMA

    def validate(
        self,
        *,
        stream_key_width: int,
        state_width: int,
        intention_width: int,
        context_width: int,
    ) -> ExternalMultiStreamTransitionContextResult:
        if self.schema != EXTERNAL_MULTI_STREAM_TRANSITION_ROUTER_SCHEMA:
            raise ValueError("unsupported multi-stream transition result schema")
        _validate_stream_key(self.stream_key, width=stream_key_width)
        self.result.validate(
            state_width=state_width,
            intention_width=intention_width,
            context_width=context_width,
        )
        return self


class ExternalMultiStreamTransitionContextRouter:
    """Keep interleaved stream evidence separate over one factual bank.

    ``stream_key`` is an opaque binding token, potentially produced by a
    learned upstream identity mechanism. It is used only to isolate transport
    state; it is not a task label, simulator identifier, or reasoning branch.
    All stream-local routers share one model bank, context encoder, sparse
    evidence index, route query, and verifier boundary.
    """

    schema = EXTERNAL_MULTI_STREAM_TRANSITION_ROUTER_SCHEMA

    def __init__(
        self,
        router: ExternalOnlineTransitionContextRouter,
        *,
        stream_key_width: int,
    ) -> None:
        if not isinstance(router, ExternalOnlineTransitionContextRouter):
            raise TypeError("multi-stream router requires an online transition router")
        if stream_key_width < 1:
            raise ValueError("multi-stream key width must be positive")
        self.router = router
        self.stream_key_width = int(stream_key_width)
        self._streams: dict[
            tuple[float, ...], ExternalOnlineTransitionContextRouter
        ] = {}
        self._bound_slot_ids: dict[tuple[float, ...], int] = {}

    @property
    def bank(self) -> ExternalTransitionModelBank:
        """Return the one shared factual model bank."""

        return self.router.bank

    @property
    def stream_count(self) -> int:
        return len(self._streams)

    @property
    def stream_keys(self) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor(key, dtype=torch.float32) for key in self._streams)

    @property
    def provisional_candidate_count(self) -> int:
        return sum(child.provisional_candidate_count for child in self._streams.values())

    def configuration(self) -> dict[str, object]:
        """Describe the shared/isolated ownership boundary."""

        return {
            "schema": self.schema,
            "stream_key_width": self.stream_key_width,
            "identity": "normalized_opaque_stream_key_v1",
            "shared": (
                "bank_context_encoder_route_query_sparse_evidence_evaluator_cost_ledger"
            ),
            "isolated": (
                "pending_active_slot_quarantine_address_adapter_candidates"
            ),
        }

    def _stream_id(self, stream_key: torch.Tensor) -> tuple[float, ...]:
        normalized = _validate_stream_key(stream_key, width=self.stream_key_width)
        return tuple(round(float(value), 7) for value in normalized.tolist())

    def _child(
        self,
        stream_key: torch.Tensor,
    ) -> tuple[tuple[float, ...], ExternalOnlineTransitionContextRouter]:
        stream_id = self._stream_id(stream_key)
        child = self._streams.get(stream_id)
        if child is None:
            child = self.router.fork_stream()
            self._streams[stream_id] = child
        return stream_id, child

    def _refresh_children(self) -> None:
        for child in self._streams.values():
            child._refresh_active_slot()
        for stream_id, slot_id in tuple(self._bound_slot_ids.items()):
            try:
                self.bank.physical_index_for_slot_id(slot_id)
            except KeyError:
                del self._bound_slot_ids[stream_id]

    def pending_observations(self, stream_key: torch.Tensor) -> int:
        stream_id = self._stream_id(stream_key)
        child = self._streams.get(stream_id)
        return 0 if child is None else child.pending_observations

    def quarantined_observations(self, stream_key: torch.Tensor) -> int:
        stream_id = self._stream_id(stream_key)
        child = self._streams.get(stream_id)
        return 0 if child is None else child.quarantined_observations

    def bound_slot_id(self, stream_key: torch.Tensor) -> int | None:
        """Return the stable factual slot currently preferred by one stream."""

        stream_id = self._stream_id(stream_key)
        return self._bound_slot_ids.get(stream_id)

    def provisional_model_at(
        self,
        stream_key: torch.Tensor,
        candidate_index: int = 0,
    ) -> nn.Module:
        _, child = self._child(stream_key)
        return child.provisional_model_at(candidate_index)

    def provisional_context_at(
        self,
        stream_key: torch.Tensor,
        candidate_index: int = 0,
    ) -> torch.Tensor:
        _, child = self._child(stream_key)
        return child.provisional_context_at(candidate_index)

    def provisional_evidence_count(
        self,
        stream_key: torch.Tensor,
        candidate_index: int = 0,
    ) -> int:
        _, child = self._child(stream_key)
        return child.provisional_evidence_count(candidate_index)

    def observe(
        self,
        observation: ExternalTransitionObservation,
        stream_key: torch.Tensor,
    ) -> ExternalMultiStreamTransitionContextResult:
        normalized = _validate_stream_key(stream_key, width=self.stream_key_width)
        stream_id, child = self._child(normalized)
        result = child.observe(
            observation,
            preferred_slot_id=self._bound_slot_ids.get(stream_id),
        )
        if result.stable_slot_id is not None and result.status in {
            "matched",
            "sparse_matched",
            "continuation",
        }:
            self._bound_slot_ids[stream_id] = result.stable_slot_id
        return ExternalMultiStreamTransitionContextResult(
            stream_key=normalized,
            result=result,
        ).validate(
            stream_key_width=self.stream_key_width,
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )

    def adaptation_step(
        self,
        result: ExternalMultiStreamTransitionContextResult,
        optimizer: torch.optim.Optimizer | Mapping[str, torch.optim.Optimizer] | None,
        *,
        replay_evidence: bool = True,
    ) -> float:
        result.validate(
            stream_key_width=self.stream_key_width,
            state_width=self.bank.state_width,
            intention_width=self.bank.intention_width,
            context_width=self.bank.context_width,
        )
        stream_id = self._stream_id(result.stream_key)
        child = self._streams.get(stream_id)
        if child is None:
            raise KeyError("multi-stream result belongs to an unknown stream")
        return child.adaptation_step(
            result.result,
            optimizer,
            replay_evidence=replay_evidence,
        )

    def promote_staged_candidate(
        self,
        stream_key: torch.Tensor,
        heldout_observation: ExternalTransitionObservation,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
        **kwargs: Any,
    ) -> ExternalTransitionModelCandidateReceipt:
        _, child = self._child(stream_key)
        receipt = child.promote_staged_candidate(
            heldout_observation,
            retention_probe,
            **kwargs,
        )
        if receipt.accepted and receipt.slot_id is not None:
            stream_id = self._stream_id(stream_key)
            self._bound_slot_ids[stream_id] = receipt.slot_id
        self._refresh_children()
        return receipt

    def grow_verified(
        self,
        destination_capacity: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> object:
        receipt = self.router.grow_verified(destination_capacity, retention_probe)
        self._refresh_children()
        return receipt

    def evict_verified_id(
        self,
        slot_id: int,
        retention_probe: Callable[[ExternalTransitionModelBank], bool],
    ) -> object:
        receipt = self.router.evict_verified_id(slot_id, retention_probe)
        self._refresh_children()
        return receipt

    def consolidate_verified(
        self,
        first_slot_id: int,
        second_slot_id: int,
        heldout: Sequence[ExternalTransitionObservation],
        *,
        prediction_tolerance: float = 1e-6,
        retention_probe: Callable[
            [ExternalMultiStreamTransitionContextRouter], bool
        ]
        | None = None,
    ) -> ExternalTransitionModelConsolidationReceipt:
        """Share equivalent factual slots on an isolated router copy.

        Stream addresses and context keys remain distinct.  Only the physical
        model object is shared, and the complete stream-local router state is
        committed together with that alias.  A retention probe is read-only:
        mutating a candidate during the probe rejects the transaction.
        """

        for name, value in (
            ("first factual slot ID", first_slot_id),
            ("second factual slot ID", second_slot_id),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"multi-stream {name} is invalid")
        self.bank.physical_index_for_slot_id(first_slot_id)
        self.bank.physical_index_for_slot_id(second_slot_id)
        if not isinstance(heldout, Sequence) or isinstance(heldout, (str, bytes)):
            raise TypeError("multi-stream consolidation held-out evidence is invalid")
        if not heldout:
            raise ValueError("multi-stream consolidation needs held-out evidence")
        if prediction_tolerance < 0.0:
            raise ValueError("multi-stream consolidation tolerance cannot be negative")
        if retention_probe is not None and not callable(retention_probe):
            raise TypeError("multi-stream consolidation retention probe is invalid")

        candidate = type(self).from_payload(self.state_payload())

        def candidate_retention_probe(
            _bank: ExternalTransitionModelBank,
        ) -> bool:
            candidate._refresh_children()
            before = candidate.digest()
            accepted = (
                True
                if retention_probe is None
                else bool(retention_probe(candidate))
            )
            return accepted and candidate.digest() == before

        receipt = candidate.router.bank.consolidate_verified(
            candidate.router.bank.physical_index_for_slot_id(first_slot_id),
            candidate.router.bank.physical_index_for_slot_id(second_slot_id),
            heldout,
            prediction_tolerance=prediction_tolerance,
            retention_probe=candidate_retention_probe,
        )
        if not receipt.accepted:
            return receipt
        candidate._refresh_children()
        self.router = candidate.router
        self._streams = candidate._streams
        self._bound_slot_ids = candidate._bound_slot_ids
        return receipt

    def state_payload(self) -> dict[str, object]:
        """Serialize shared memory once and stream-local transient state."""

        # Persist the logical slot address and its current physical cache
        # together.  A child can retain only ``active_slot_id`` after a bank
        # eviction/replacement; normalizing the physical index before taking
        # the snapshot keeps save/load byte-exact.
        self._refresh_children()
        shared = self.router.fork_stream().state_payload()
        transient_keys = (
            "address_adapter",
            "pending",
            "ambiguous_quarantine",
            "active_slot",
            "active_slot_id",
            "conflict_windows",
            "provisional_candidates",
            "provisional_context",
            "provisional_model",
            "provisional_model_family",
            "provisional_observations",
        )
        streams: list[dict[str, object]] = []
        for stream_id, child in self._streams.items():
            child_payload = child.state_payload()
            streams.append(
                {
                    "stream_key": list(stream_id),
                    "bound_slot_id": self._bound_slot_ids.get(stream_id),
                    "state": {
                        key: copy.deepcopy(child_payload[key]) for key in transient_keys
                    },
                }
            )
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "shared_router": shared,
            "streams": streams,
            "sha256": _state_digest(shared, streams),
        }

    def digest(self) -> str:
        """Return the checksum of the complete shared plus stream-local state."""

        return str(self.state_payload()["sha256"])

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        evidence_evaluator: nn.Module | None = None,
        prior_selection_probe: Callable[
            [nn.Module, nn.Module, ExternalTransitionObservation], tuple[float, float]
        ]
        | None = None,
    ) -> ExternalMultiStreamTransitionContextRouter:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported multi-stream transition-router payload")
        configuration = payload.get("configuration")
        shared_payload = payload.get("shared_router")
        streams = payload.get("streams")
        if not isinstance(configuration, Mapping) or not isinstance(
            shared_payload, Mapping
        ) or not isinstance(streams, list):
            raise TypeError("multi-stream transition-router payload is incomplete")
        stream_key_width = int(configuration["stream_key_width"])
        expected_configuration = {
            "schema": cls.schema,
            "stream_key_width": stream_key_width,
            "identity": "normalized_opaque_stream_key_v1",
            "shared": (
                "bank_context_encoder_route_query_sparse_evidence_evaluator_cost_ledger"
            ),
            "isolated": (
                "pending_active_slot_quarantine_address_adapter_candidates"
            ),
        }
        if dict(configuration) != expected_configuration:
            raise ValueError("multi-stream transition-router configuration mismatch")
        base = ExternalOnlineTransitionContextRouter.from_payload(
            shared_payload,
            evidence_evaluator=evidence_evaluator,
            prior_selection_probe=prior_selection_probe,
        )
        router = cls(base, stream_key_width=stream_key_width)
        if payload.get("sha256") != _state_digest(shared_payload, streams):
            raise ValueError("multi-stream transition-router checksum mismatch")
        for item in streams:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("state"), Mapping
            ):
                raise TypeError("multi-stream stream state is invalid")
            stream_key = torch.tensor(item["stream_key"], dtype=torch.float32)
            _validate_stream_key(stream_key, width=stream_key_width)
            # The serialized key is already the rounded canonical stream ID.
            # Normalizing it a second time can move the final float32 bit and
            # make an otherwise valid payload fail exact round-trip checks.
            stream_id = tuple(round(float(value), 7) for value in stream_key.tolist())
            if stream_id in router._streams:
                raise ValueError("multi-stream stream keys are duplicated")
            bound_slot_id = item.get("bound_slot_id")
            if bound_slot_id is not None:
                if (
                    not isinstance(bound_slot_id, int)
                    or isinstance(bound_slot_id, bool)
                    or bound_slot_id < 0
                ):
                    raise ValueError("multi-stream bound slot ID is invalid")
                try:
                    base.bank.physical_index_for_slot_id(bound_slot_id)
                except KeyError as error:
                    raise ValueError(
                        "multi-stream bound slot ID is unknown"
                    ) from error
            full_payload = copy.deepcopy(dict(shared_payload))
            full_payload.update(copy.deepcopy(dict(item["state"])))
            child = ExternalOnlineTransitionContextRouter.from_payload(
                full_payload,
                evidence_evaluator=evidence_evaluator,
                prior_selection_probe=prior_selection_probe,
            )
            child.bank = base.bank
            child.context_encoder = base.context_encoder
            child.route_query = base.route_query
            child.sparse_evidence = base.sparse_evidence
            child.evidence_evaluator = base.evidence_evaluator
            child.prior_selection_cost_ledger = base.prior_selection_cost_ledger
            router._streams[stream_id] = child
            if bound_slot_id is not None:
                router._bound_slot_ids[stream_id] = bound_slot_id
        router._refresh_children()
        return router
