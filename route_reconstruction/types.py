"""Shared data structures for the PathSnap route reconstruction module.

These structures are the initial module contract. Adapters can be added once
upstream candidate-generation and API schemas are agreed by the team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


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


@dataclass(frozen=True)
class RoadArc:
    """A directed road segment. The arc ID is its row index in a ``.arcs`` file."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float | None = None


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
    ``classifier_probability`` is an optional score from the ML module.
    """

    arc_id: int
    distance_m: float
    fraction_along: float | None = None
    classifier_probability: float | None = None


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


@dataclass(frozen=True)
class MapMatchingResult:
    """Decoder output aligned to input observations."""

    matched_arc_ids: Sequence[int | None]
    route_arc_ids: Sequence[int]
    log_probability: float | None
    diagnostics: Sequence[str] = field(default_factory=tuple)
    segment_ids: Sequence[int | None] = field(default_factory=tuple)
