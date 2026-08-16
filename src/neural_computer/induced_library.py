"""An external program store that grows without bound and never overwrites.

The temporal bank is a fixed-capacity, router-addressed store of prototype
artifacts, and it is the right shape for the family it holds. It is the wrong
shape for programs induced from feedback: those are counter machines rather
than prototype matches, there is no reason for their number to be bounded by a
constant chosen in advance, and the whole point of inducing them is that the
set is open.

Three properties are load-bearing here, and each is a mechanism rather than an
aspiration.

**Additive.** `append` is the only way in. There is no eviction, no overwrite,
and no slot reuse, so admitting capability N+1 cannot damage capability N. The
digest covers every record in order, so a load that succeeds proves nothing
earlier changed.

**Unbounded.** Capacity is not a constructor argument. The store is a list on
disk; growing it does not resize any network, and no controller parameter
depends on how many records exist.

**Selectively addressed.** Recognition must not require executing every record.
Each record carries a `signature`: the presses its program makes on a canonical
symbol stream fixed by the alphabet size alone. Finding candidates is then a
Hamming comparison against a stored bit vector, and only the closest few are
ever executed. That is what makes the cost of consulting a large library close
to the cost of consulting a small one, which is the property that has to hold
if a library is to be worth accumulating at all.

The store holds programs, not hypotheses. A record is executable on its own
terms -- program, initial counters, alphabet width -- and whatever produced it
lives in an opaque `provenance` mapping that this module never interprets.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .control_flow import ControlFlowProgram

INDUCED_PROGRAM_LIBRARY_SCHEMA = "neural-computer.induced-program-library.v1"
INDUCED_PROGRAM_RECORD_SCHEMA = "neural-computer.induced-program-record.v1"
INDUCED_LIBRARY_EXTENSION = ".library"

# The canonical stream every signature is taken on. Fixed by the alphabet size
# alone so two records over the same alphabet are always comparable, and long
# enough that machines differing only deep in their state graph separate.
SIGNATURE_LENGTH = 256
SIGNATURE_SEED = 0x5EED


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    Path(temporary_name).replace(path)


def canonical_signature_stream(alphabet: int, *, length: int = SIGNATURE_LENGTH) -> tuple[int, ...]:
    """The stream signatures are taken on, derived from the alphabet alone.

    Deliberately not drawn from the environment: a signature has to be
    comparable across records admitted at different times, under different
    frontends, from different tasks. A stream that depends on any of those
    would make two records incomparable for reasons that have nothing to do
    with what they compute.
    """

    if alphabet < 2:
        raise ValueError("a signature stream needs at least two symbols")
    if length < 1:
        raise ValueError("a signature stream needs at least one step")
    # A linear congruential walk, so the stream is reproducible without pulling
    # in a tensor library or depending on any global RNG state.
    value = SIGNATURE_SEED
    stream: list[int] = []
    for _ in range(int(length)):
        value = (value * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        stream.append((value >> 33) % int(alphabet))
    return tuple(stream)


@dataclass(frozen=True)
class InducedProgramRecord:
    """One executable program, its start state, and how it was arrived at."""

    program: ControlFlowProgram
    initial_counters: tuple[int, ...]
    alphabet: int
    signature: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    # How many answers this program can give. Two -- press or not -- is what
    # every record written before answers were a choice holds, and what this
    # still defaults to, so those files load and digest exactly as they did.
    action_count: int = 2
    schema: str = INDUCED_PROGRAM_RECORD_SCHEMA

    def validate(self) -> InducedProgramRecord:
        if self.schema != INDUCED_PROGRAM_RECORD_SCHEMA:
            raise ValueError("unsupported induced program record schema")
        self.program.validate()
        if len(self.initial_counters) != self.program.counter_count:
            raise ValueError("initial counters do not match the program")
        if any(value < 0 for value in self.initial_counters):
            raise ValueError("initial counters must be non-negative")
        if self.alphabet < 2:
            raise ValueError("an induced program needs at least two symbols")
        if not self.signature:
            raise ValueError("an induced program record needs a signature")
        if self.action_count < 2:
            raise ValueError("an induced program needs at least two actions")
        if any(not 0 <= int(bit) < self.action_count for bit in self.signature):
            raise ValueError("a signature step is outside the action set")
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a mapping")
        return self

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema": self.schema,
            "program": self.program.payload(),
            "initial_counters": list(self.initial_counters),
            "alphabet": int(self.alphabet),
            "signature": list(self.signature),
            "provenance": json.loads(json.dumps(self.provenance, sort_keys=True)),
        }
        # Written only when it is not the binary default, so every record and
        # every library digest recorded before answers were a choice is
        # byte-identical to what it was.
        if self.action_count != 2:
            payload["action_count"] = self.action_count
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> InducedProgramRecord:
        if not isinstance(payload, dict):
            raise TypeError("induced program record payload must be a mapping")
        if payload.get("schema") != INDUCED_PROGRAM_RECORD_SCHEMA:
            raise ValueError("unsupported induced program record schema")
        return cls(
            program=ControlFlowProgram.from_payload(payload.get("program")),
            initial_counters=tuple(int(value) for value in payload["initial_counters"]),
            alphabet=int(payload["alphabet"]),
            signature=tuple(int(bit) for bit in payload["signature"]),
            provenance=dict(payload.get("provenance") or {}),
            action_count=int(payload.get("action_count", 2)),
        ).validate()

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def signature_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Hamming distance, the cheap test that precedes any execution."""

    if len(left) != len(right):
        raise ValueError("signatures over different streams are not comparable")
    return sum(1 for a, b in zip(left, right) if int(a) != int(b))


class InducedProgramLibrary:
    """Append-only, checksummed, signature-indexed external program store."""

    schema = INDUCED_PROGRAM_LIBRARY_SCHEMA

    def __init__(self, *, alphabet: int, frontend_digest: str | None = None) -> None:
        if alphabet < 2:
            raise ValueError("an induced program library needs at least two symbols")
        if frontend_digest is not None and len(frontend_digest) != 64:
            raise ValueError("frontend digest must be a SHA-256 hex digest")
        self.alphabet = int(alphabet)
        self.frontend_digest = frontend_digest
        self._records: list[InducedProgramRecord] = []

    @property
    def record_count(self) -> int:
        return len(self._records)

    def record(self, slot: int) -> InducedProgramRecord:
        if not 0 <= slot < self.record_count:
            raise IndexError("induced program library slot is out of range")
        return self._records[slot]

    def records(self) -> tuple[InducedProgramRecord, ...]:
        return tuple(self._records)

    def append(self, record: InducedProgramRecord) -> int:
        """The only way in. Never replaces, never evicts, never reorders."""

        record.validate()
        if record.alphabet != self.alphabet:
            raise ValueError("induced program alphabet does not match the library")
        expected = len(canonical_signature_stream(self.alphabet))
        if len(record.signature) != expected:
            raise ValueError("induced program signature is over the wrong stream")
        self._records.append(record)
        return len(self._records) - 1

    def duplicate_of(self, signature: Sequence[int]) -> int | None:
        """The slot a candidate would merely restate, if there is one.

        Behavioural rather than syntactic: two differently compiled programs
        that press identically on the canonical stream are the same capability
        as far as this store is concerned, and admitting the second one would
        lengthen the library without shortening anything.
        """

        for slot, record in enumerate(self._records):
            if signature_distance(record.signature, signature) == 0:
                return slot
        return None

    def nearest(
        self, signature: Sequence[int], *, limit: int = 4
    ) -> tuple[tuple[int, int], ...]:
        """The closest records by signature: (slot, distance), nearest first.

        This is the whole reason recognition does not cost a pass over the
        library. Comparing bit vectors is not free, but it is not an episode,
        and only `limit` records are ever executed afterwards.
        """

        if limit < 1:
            raise ValueError("nearest needs a positive limit")
        scored = [
            (slot, signature_distance(record.signature, signature))
            for slot, record in enumerate(self._records)
        ]
        scored.sort(key=lambda item: (item[1], item[0]))
        return tuple(scored[: int(limit)])

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alphabet": self.alphabet,
            "frontend_digest": self.frontend_digest,
            "signature_length": SIGNATURE_LENGTH,
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode()
        )
        # Order is part of the identity: an append-only store that reordered
        # would not be append-only.
        for record in self._records:
            digest.update(record.digest().encode())
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "records": [record.payload() for record in self._records],
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> InducedProgramLibrary:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported induced program library schema")
        configuration = payload.get("configuration")
        records = payload.get("records")
        if not isinstance(configuration, dict) or not isinstance(records, list):
            raise TypeError("induced program library payload is malformed")
        library = cls(
            alphabet=int(configuration["alphabet"]),
            frontend_digest=configuration.get("frontend_digest"),
        )
        if library.configuration() != configuration:
            raise ValueError("induced program library configuration mismatch")
        for item in records:
            library.append(InducedProgramRecord.from_payload(item))
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != library.digest():
            raise ValueError("induced program library checksum mismatch")
        return library

    def save(self, path: Path) -> None:
        """Atomically persist, with an independent whole-file checksum."""

        path = Path(path)
        if path.suffix != INDUCED_LIBRARY_EXTENSION:
            raise ValueError(
                f"induced program libraries must use the "
                f"{INDUCED_LIBRARY_EXTENSION} extension"
            )
        _atomic_text_write(
            path, json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        )
        _atomic_text_write(
            path.with_suffix(path.suffix + ".sha256"), _sha256_file(path) + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> InducedProgramLibrary:
        path = Path(path)
        if path.suffix != INDUCED_LIBRARY_EXTENSION:
            raise ValueError(
                f"induced program libraries must use the "
                f"{INDUCED_LIBRARY_EXTENSION} extension"
            )
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise ValueError("induced program library checksum is missing")
        if checksum_path.read_text().strip() != _sha256_file(path):
            raise ValueError("induced program library file checksum mismatch")
        return cls.from_payload(json.loads(path.read_text()))


__all__ = [
    "INDUCED_LIBRARY_EXTENSION",
    "INDUCED_PROGRAM_LIBRARY_SCHEMA",
    "INDUCED_PROGRAM_RECORD_SCHEMA",
    "InducedProgramLibrary",
    "InducedProgramRecord",
    "canonical_signature_stream",
    "signature_distance",
]
