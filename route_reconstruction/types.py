"""Shared data structures for the PathSnap route reconstruction module.

These structures are the initial module contract. Adapters can be added once
upstream candidate-generation and API schemas are agreed by the team.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, Sequence


def _finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_id(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class GPSObservation:
    """One ordered GPS sample; coordinates are longitude/latitude in degrees."""

    timestamp_s: float | None
    longitude: float | None
    latitude: float | None


@dataclass(frozen=True)
class RoadNode:
    """A road-network node. The node ID is the row index in a ``.nodes`` file."""

    node_id: int
    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        _validate_id(self.node_id, "node_id")
        if not _finite_real(self.longitude) or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("node longitude must be finite and in [-180, 180]")
        if not _finite_real(self.latitude) or not -90.0 <= self.latitude <= 90.0:
            raise ValueError("node latitude must be finite and in [-90, 90]")


@dataclass(frozen=True)
class RoadArc:
    """A directed road segment. The arc ID is its row index in a ``.arcs`` file."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float | None = None

    def __post_init__(self) -> None:
        _validate_id(self.arc_id, "arc_id")
        _validate_id(self.from_node_id, "from_node_id")
        _validate_id(self.to_node_id, "to_node_id")
        if self.length_m is not None and (
            not _finite_real(self.length_m) or self.length_m < 0.0
        ):
            raise ValueError("length_m must be finite and non-negative")


@dataclass(frozen=True)
class RoadNetwork:
    """Directed graph keyed by node and arc IDs."""

    nodes: Mapping[int, RoadNode]
    arcs: Mapping[int, RoadArc]


@dataclass(frozen=True)
class CandidateState:
    """A road candidate for one observation.

    ``distance_m`` is the observation-to-road distance used by the geometric
    emission model. ``fraction_along`` is the candidate's position along the
    directed arc in [0, 1], when the candidate generator can provide it.
    ``classifier_probability`` is an optional calibrated probability from the
    ML module.
    """

    arc_id: int
    distance_m: float
    fraction_along: float | None = None
    classifier_probability: float | None = None

    def __post_init__(self) -> None:
        _validate_id(self.arc_id, "arc_id")
        if not _finite_real(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("distance_m must be finite and non-negative")
        if self.fraction_along is not None and (
            not _finite_real(self.fraction_along)
            or not 0.0 <= self.fraction_along <= 1.0
        ):
            raise ValueError("fraction_along must be finite and in [0, 1]")
        if self.classifier_probability is not None and (
            not _finite_real(self.classifier_probability)
            or not 0.0 <= self.classifier_probability <= 1.0
        ):
            raise ValueError("classifier_probability must be finite and in [0, 1]")


@dataclass(frozen=True)
class MapMatchingInput:
    """Inputs for matching one trip, in chronological observation order."""

    observations: Sequence[GPSObservation]
    candidates_by_observation: Sequence[Sequence[CandidateState]]
    network: RoadNetwork


@dataclass(frozen=True)
class MapMatchingConfig:
    """Configurable physical scales used by the decoder, in metric units."""

    emission_sigma_m: float = 20.0
    transition_beta_m: float = 50.0
    max_speed_mps: float = 55.6
    max_gap_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in ("emission_sigma_m", "transition_beta_m", "max_speed_mps", "max_gap_seconds"):
            value = getattr(self, name)
            if not _finite_real(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MapMatchingResult:
    """Decoder output aligned to input observations."""

    matched_arc_ids: Sequence[int | None]
    route_arc_ids: Sequence[int]
    log_score: float | None
    diagnostics: Sequence[str] = field(default_factory=tuple)
    segment_ids: Sequence[int | None] = field(default_factory=tuple)
