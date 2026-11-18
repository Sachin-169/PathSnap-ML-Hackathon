"""Transition likelihoods for consecutive GPS map-matching states."""

from __future__ import annotations

import math
from numbers import Real

from .network import RoadNetworkRouter
from .types import CandidateState


def transition_probability(
    network_distance_m: float, gps_displacement_m: float, beta_m: float
) -> float:
    """Return an exponential likelihood for route/GPS distance agreement.

    The score decreases exponentially with the absolute difference between
    routed distance and observed displacement. ``beta_m`` is the scale of
    tolerated disagreement and should be selected on development trips.
    """
    values = (network_distance_m, gps_displacement_m, beta_m)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise ValueError("transition distances and beta_m must be real numbers")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("transition distances and beta_m must be finite")
    if network_distance_m < 0 or gps_displacement_m < 0:
        raise ValueError("transition distances must be non-negative")
    if beta_m <= 0:
        raise ValueError("beta_m must be positive")
    return math.exp(-abs(network_distance_m - gps_displacement_m) / beta_m)


def candidate_transition_probability(
    previous: CandidateState,
    current: CandidateState,
    gps_displacement_m: float,
    beta_m: float,
    router: RoadNetworkRouter,
) -> float:
    """Score a candidate pair; return zero when directed roads are disconnected."""
    network_distance = router.candidate_distance_m(previous, current)
    if network_distance is None:
        return 0.0
    return transition_probability(network_distance, gps_displacement_m, beta_m)
