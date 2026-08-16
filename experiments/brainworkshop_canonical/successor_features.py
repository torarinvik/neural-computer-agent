"""What a policy visits, kept separately from how much that is worth.

The navigation model was per-task and thrown away. It had to be: a plan is a
route to *one* goal, so a new goal meant a new search, and nothing accumulated
even though the dynamics never changed.

Successor features are DeepMind's answer to exactly this (Barreto et al., 2017;
PNAS 2020). Instead of a value, a policy stores the discounted future
occupancy of *features* -- how much of each thing it goes on to see. A task is
then a weight vector over those features, value is one dot product, and

    Q^pi_w(s, a) = psi^pi(s, a) . w

evaluates any stored policy against any reward the features can express,
instantly and without a single further step in the world. Generalised policy
improvement then acts greedily across the whole stored set,

    a* = argmax_a max_i  psi^i(s, a) . w

which is guaranteed to be at least as good as the best single stored policy and
is usually better, because it stitches: follow one policy while it is the best
one to follow, then switch.

Three things make this the right import rather than an aspirational one.

**It is exact here, and needs no gradients.** Every published version
approximates psi with a network because the state space is large. This world
has eight places and a handful of actions, so psi is the solution of a linear
system -- `(I - gamma P)^-1 P` -- computed in closed form. Nothing is trained,
nothing is fitted, and the value it reports is the return, not an estimate of
one.

**The cumulant is already there.** phi is the indicator of the place arrived
at. A one-hot weight vector reproduces the goal-reaching task exactly, so this
is a strict generalisation of the previous experiment rather than a different
one, and the earlier number is available as a check.

**The goal language widens for free.** Once a task is a vector, a conjunction
of places is a two-hot vector, avoidance is a negative entry, and preference is
a graded one. None of those are expressible as "the place to walk to", and none
of them cost new machinery.

The library at the bottom is the accumulation half. It is append-only and
checksummed for the same reasons `induced_library` is: admitting policy N+1
must not be able to damage policy N, and a load that succeeds proves nothing
earlier moved.
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

import torch

from neural_computer.promotion import sha256_file

from .world_model import WorldModel

SUCCESSOR_FEATURES_SCHEMA = "neural-computer.successor-features.v1"
SUCCESSOR_LIBRARY_SCHEMA = "neural-computer.successor-feature-library.v1"
SUCCESSOR_RECORD_SCHEMA = "neural-computer.successor-feature-record.v1"
SUCCESSOR_LIBRARY_EXTENSION = ".successors"

DEFAULT_DISCOUNT = 0.95
# Stored to nine places so that two runs that computed the same psi write the
# same bytes. The solve is float64; the digest should not depend on its last
# bit.
STORED_PRECISION = 9


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


def known_successor(model: WorldModel, place: int, action: int) -> int:
    """Where the model believes this leads, or here if it has no evidence.

    An untried action is not a self-loop in reality, and `plan_to` is right to
    refuse to route through one. A *policy*, though, has to say something at
    every place, and "stay put" is the choice that claims least: it neither
    invents an edge nor lets the value function collect reward it has no reason
    to expect.
    """

    following = model.successor(place, action)
    return int(place) if following is None else int(following)


def transition_matrix(model: WorldModel) -> list[list[int]]:
    """`[action][place] -> place`, with unknown cells held in place."""

    return [
        [known_successor(model, place, action) for place in range(model.place_count)]
        for action in range(model.action_count)
    ]


def place_cumulants(place_count: int) -> torch.Tensor:
    """phi: the indicator of the place arrived at.

    The simplest cumulant that makes "go there" a weight vector. Everything the
    goal language gains later -- conjunction, avoidance, preference -- is a
    different `w` over this same phi, not a different phi.
    """

    return torch.eye(int(place_count), dtype=torch.float64)


def successor_features(
    model: WorldModel,
    policy: Sequence[int],
    *,
    discount: float = DEFAULT_DISCOUNT,
    cumulants: torch.Tensor | None = None,
) -> torch.Tensor:
    """psi[place][action] for one deterministic policy, in closed form.

    psi(s, a) = phi(s') + gamma * m(s'), where m(s) is the discounted feature
    occupancy of following the policy from s. m satisfies m = P phi + gamma P m
    for the policy's transition matrix P, so m = (I - gamma P)^-1 P phi and no
    iteration is involved -- the number returned is the return.
    """

    if not 0.0 <= discount < 1.0:
        raise ValueError("the discount must be in [0, 1)")
    places = int(model.place_count)
    actions = int(model.action_count)
    if len(policy) != places:
        raise ValueError("a policy must choose an action at every place")
    if any(not 0 <= int(action) < actions for action in policy):
        raise ValueError("a policy chose an action outside the protocol")
    features = place_cumulants(places) if cumulants is None else cumulants.double()
    if features.shape[0] != places:
        raise ValueError("cumulants must be defined at every place")

    table = transition_matrix(model)
    following = torch.zeros((places, places), dtype=torch.float64)
    for place in range(places):
        following[place, table[int(policy[place])][place]] = 1.0

    # m = (I - gamma P)^-1 P phi
    identity = torch.eye(places, dtype=torch.float64)
    occupancy = torch.linalg.solve(
        identity - discount * following, following @ features
    )

    psi = torch.zeros((places, actions, features.shape[1]), dtype=torch.float64)
    for place in range(places):
        for action in range(actions):
            arrived = table[action][place]
            psi[place, action] = features[arrived] + discount * occupancy[arrived]
    return psi


def greedy_policy(
    model: WorldModel,
    weights: torch.Tensor,
    *,
    discount: float = DEFAULT_DISCOUNT,
    cumulants: torch.Tensor | None = None,
    sweeps: int = 512,
) -> tuple[int, ...]:
    """The best policy the *model* supports for this weight vector.

    Value iteration over the believed dynamics. Used for two different things
    and it is worth keeping them apart: to manufacture the base policies that
    get stored, and, at evaluation time, as the control that re-solves from
    scratch for every new task -- which is precisely the cost successor
    features are supposed to remove.
    """

    places = int(model.place_count)
    actions = int(model.action_count)
    features = place_cumulants(places) if cumulants is None else cumulants.double()
    reward = (features @ weights.double()).tolist()
    table = transition_matrix(model)

    value = [0.0] * places
    for _ in range(int(sweeps)):
        updated = [
            max(
                reward[table[action][place]] + discount * value[table[action][place]]
                for action in range(actions)
            )
            for place in range(places)
        ]
        if max(abs(a - b) for a, b in zip(updated, value)) < 1e-12:
            value = updated
            break
        value = updated
    return tuple(
        max(
            range(actions),
            key=lambda action: (
                reward[table[action][place]] + discount * value[table[action][place]]
            ),
        )
        for place in range(places)
    )


def generalised_policy_improvement(
    psis: Sequence[torch.Tensor], place: int, weights: torch.Tensor
) -> int:
    """argmax over actions of the best stored policy's value for this task.

    The whole transfer claim is in this line. Nothing here consults a model,
    replans, or takes a step in the world: the stored occupancies are dotted
    with a weight vector that may never have been seen before, and the action
    that wins is played.
    """

    if not psis:
        raise ValueError("generalised policy improvement needs a stored policy")
    weights = weights.double()
    scores = torch.stack([psi[int(place)] @ weights for psi in psis])
    return int(torch.argmax(scores.max(dim=0).values).item())


def gpi_policy(
    psis: Sequence[torch.Tensor], place_count: int, weights: torch.Tensor
) -> tuple[int, ...]:
    """The stitched policy, as a thing that can itself be evaluated."""

    return tuple(
        generalised_policy_improvement(psis, place, weights)
        for place in range(int(place_count))
    )


def policy_values(psi: torch.Tensor, policy: Sequence[int], weights: torch.Tensor):
    """What following this policy from each place is worth under this task."""

    weights = weights.double()
    return [
        float(psi[place, int(policy[place])] @ weights)
        for place in range(psi.shape[0])
    ]


def stitching_gain(
    model: WorldModel,
    psis: Sequence[torch.Tensor],
    policies: Sequence[Sequence[int]],
    weights: torch.Tensor,
    *,
    discount: float = DEFAULT_DISCOUNT,
) -> dict[str, Any]:
    """How much stitching buys over the best single stored policy.

    The guarantee is that generalised policy improvement is no worse than every
    stored policy, so a positive number is the only evidence the *set* does
    something no member does. It has to be measured as the value of the
    stitched policy, not as a maximum over stored values at one state: those
    maxima are the same number by construction, and a first version compared
    them and reported a gain of exactly zero everywhere -- which is what a
    tautology looks like from the outside.
    """

    stitched = gpi_policy(psis, model.place_count, weights)
    stitched_values = policy_values(
        successor_features(model, stitched, discount=discount), stitched, weights
    )
    best_single = [
        max(policy_values(psi, policy, weights)[place] for psi, policy in zip(psis, policies))
        for place in range(model.place_count)
    ]
    gains = [a - b for a, b in zip(stitched_values, best_single)]
    return {
        "policy": stitched,
        "stitched": stitched_values,
        "best_single": best_single,
        "gains": gains,
        "mean_gain": sum(gains) / len(gains) if gains else 0.0,
        "places_improved": sum(1 for gain in gains if gain > 1e-9),
    }


# --- weight vectors, which are what a task now is --------------------------


def reach(place_count: int, place: int) -> torch.Tensor:
    """Be at that place. The task the previous experiment could express."""

    weights = torch.zeros(int(place_count), dtype=torch.float64)
    weights[int(place)] = 1.0
    return weights


def reach_any(place_count: int, places: Sequence[int]) -> torch.Tensor:
    """Be at any of those places. A disjunction, which it could not."""

    weights = torch.zeros(int(place_count), dtype=torch.float64)
    for place in places:
        weights[int(place)] = 1.0
    return weights


def reach_avoiding(
    place_count: int, place: int, hazard: int, *, penalty: float = 1.0
) -> torch.Tensor:
    """Be there, and not there. A negative entry, which it could not either.

    This is the case no single goal-reaching policy is good for, and therefore
    the one worth measuring: a stored policy that walks straight through the
    hazard is optimal for its own task and wrong for this one.
    """

    weights = reach(place_count, place)
    weights[int(hazard)] -= float(penalty)
    return weights


# --- the accumulation half -------------------------------------------------


def _rounded(values) -> list:
    if isinstance(values, list):
        return [_rounded(value) for value in values]
    return round(float(values), STORED_PRECISION)


@dataclass(frozen=True)
class SuccessorFeatureRecord:
    """One policy, what it goes on to see, and where it came from."""

    policy: tuple[int, ...]
    psi: torch.Tensor
    discount: float
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = SUCCESSOR_RECORD_SCHEMA

    def validate(self) -> SuccessorFeatureRecord:
        if self.schema != SUCCESSOR_RECORD_SCHEMA:
            raise ValueError("unsupported successor feature record schema")
        if self.psi.ndim != 3:
            raise ValueError("psi must be indexed by place, action and cumulant")
        if self.psi.shape[0] != len(self.policy):
            raise ValueError("psi and the policy disagree about the place count")
        if not 0.0 <= self.discount < 1.0:
            raise ValueError("the discount must be in [0, 1)")
        if any(not 0 <= int(action) < self.psi.shape[1] for action in self.policy):
            raise ValueError("a policy chose an action outside the protocol")
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a mapping")
        return self

    @property
    def signature(self) -> tuple[int, ...]:
        """What the policy *does*, which is what makes two of them the same.

        Behavioural, exactly as in the program library: two policies derived
        from different goals that happen to act identically everywhere are one
        capability, and storing the second lengthens the library without adding
        anything to it.
        """

        return tuple(int(action) for action in self.policy)

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "policy": list(self.signature),
            "discount": round(float(self.discount), STORED_PRECISION),
            "psi": _rounded(self.psi.double().tolist()),
            "provenance": json.loads(json.dumps(self.provenance, sort_keys=True)),
        }

    @classmethod
    def from_payload(cls, payload: object) -> SuccessorFeatureRecord:
        if not isinstance(payload, dict):
            raise TypeError("successor feature record payload must be a mapping")
        if payload.get("schema") != SUCCESSOR_RECORD_SCHEMA:
            raise ValueError("unsupported successor feature record schema")
        return cls(
            policy=tuple(int(action) for action in payload["policy"]),
            psi=torch.tensor(payload["psi"], dtype=torch.float64),
            discount=float(payload["discount"]),
            provenance=dict(payload.get("provenance") or {}),
        ).validate()

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class SuccessorFeatureLibrary:
    """Append-only, checksummed store of what policies go on to see."""

    schema = SUCCESSOR_LIBRARY_SCHEMA

    def __init__(
        self,
        *,
        place_count: int,
        action_count: int,
        cumulant_dimension: int,
        frontend_digest: str | None = None,
    ) -> None:
        if place_count < 2 or action_count < 1 or cumulant_dimension < 1:
            raise ValueError("a successor feature library needs a shaped world")
        if frontend_digest is not None and len(frontend_digest) != 64:
            raise ValueError("frontend digest must be a SHA-256 hex digest")
        self.place_count = int(place_count)
        self.action_count = int(action_count)
        self.cumulant_dimension = int(cumulant_dimension)
        self.frontend_digest = frontend_digest
        self._records: list[SuccessorFeatureRecord] = []

    @property
    def record_count(self) -> int:
        return len(self._records)

    def records(self) -> tuple[SuccessorFeatureRecord, ...]:
        return tuple(self._records)

    def psis(self) -> tuple[torch.Tensor, ...]:
        return tuple(record.psi for record in self._records)

    def append(self, record: SuccessorFeatureRecord) -> int:
        record.validate()
        if record.psi.shape != (
            self.place_count,
            self.action_count,
            self.cumulant_dimension,
        ):
            raise ValueError("psi does not match the library's world")
        self._records.append(record)
        return len(self._records) - 1

    def duplicate_of(self, signature: Sequence[int]) -> int | None:
        wanted = tuple(int(action) for action in signature)
        for slot, record in enumerate(self._records):
            if record.signature == wanted:
                return slot
        return None

    def act(self, place: int, weights: torch.Tensor) -> int:
        """Generalised policy improvement over everything stored so far."""

        return generalised_policy_improvement(self.psis(), place, weights)

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "place_count": self.place_count,
            "action_count": self.action_count,
            "cumulant_dimension": self.cumulant_dimension,
            "frontend_digest": self.frontend_digest,
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode()
        )
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
    def from_payload(cls, payload: object) -> SuccessorFeatureLibrary:
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported successor feature library schema")
        configuration = payload.get("configuration")
        records = payload.get("records")
        if not isinstance(configuration, dict) or not isinstance(records, list):
            raise TypeError("successor feature library payload is malformed")
        library = cls(
            place_count=int(configuration["place_count"]),
            action_count=int(configuration["action_count"]),
            cumulant_dimension=int(configuration["cumulant_dimension"]),
            frontend_digest=configuration.get("frontend_digest"),
        )
        if library.configuration() != configuration:
            raise ValueError("successor feature library configuration mismatch")
        for item in records:
            library.append(SuccessorFeatureRecord.from_payload(item))
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != library.digest():
            raise ValueError("successor feature library checksum mismatch")
        return library

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.suffix != SUCCESSOR_LIBRARY_EXTENSION:
            raise ValueError(
                f"successor feature libraries must use the "
                f"{SUCCESSOR_LIBRARY_EXTENSION} extension"
            )
        _atomic_text_write(
            path, json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        )
        _atomic_text_write(
            path.with_suffix(path.suffix + ".sha256"), sha256_file(path) + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> SuccessorFeatureLibrary:
        path = Path(path)
        if path.suffix != SUCCESSOR_LIBRARY_EXTENSION:
            raise ValueError(
                f"successor feature libraries must use the "
                f"{SUCCESSOR_LIBRARY_EXTENSION} extension"
            )
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise ValueError("successor feature library checksum is missing")
        if checksum_path.read_text().strip() != sha256_file(path):
            raise ValueError("successor feature library file checksum mismatch")
        return cls.from_payload(json.loads(path.read_text()))


__all__ = [
    "DEFAULT_DISCOUNT",
    "SUCCESSOR_FEATURES_SCHEMA",
    "SUCCESSOR_LIBRARY_EXTENSION",
    "SUCCESSOR_LIBRARY_SCHEMA",
    "SUCCESSOR_RECORD_SCHEMA",
    "SuccessorFeatureLibrary",
    "SuccessorFeatureRecord",
    "generalised_policy_improvement",
    "gpi_policy",
    "greedy_policy",
    "known_successor",
    "place_cumulants",
    "policy_values",
    "reach",
    "reach_any",
    "reach_avoiding",
    "stitching_gain",
    "successor_features",
    "transition_matrix",
]
