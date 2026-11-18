"""Numerically stable probability calculations for the HMM decoder."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

from .network import RoadNetworkRouter
from .types import CandidateState


def log_probability(probability: float) -> float:
    """Return log(p), representing a zero probability as negative infinity."""
    if isinstance(probability, bool) or not isinstance(probability, Real) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    return -math.inf if probability == 0.0 else math.log(probability)


def log_gaussian_emission_probability(distance_m: float, sigma_m: float) -> float:
    """Gaussian emission log-density without exponentiating a small value."""
    if isinstance(distance_m, bool) or not isinstance(distance_m, Real) or not math.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be a finite non-negative value")
    if isinstance(sigma_m, bool) or not isinstance(sigma_m, Real) or not math.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("sigma_m must be a finite positive value")
    standardized_distance = distance_m / sigma_m
    return -math.log(math.sqrt(2.0 * math.pi) * sigma_m) - 0.5 * standardized_distance**2


def emission_log_probability(candidate: CandidateState, sigma_m: float) -> float:
    """Return the log emission likelihood for a candidate state."""
    if candidate.classifier_probability is not None:
        return log_probability(candidate.classifier_probability)
    return log_gaussian_emission_probability(candidate.distance_m, sigma_m)


def transition_log_probability(
    network_distance_m: float, gps_displacement_m: float, beta_m: float
) -> float:
    """Log form of the exponential route/GPS distance-consistency likelihood."""
    values = (network_distance_m, gps_displacement_m, beta_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("transition distances and beta_m must be finite")
    if network_distance_m < 0 or gps_displacement_m < 0:
        raise ValueError("transition distances must be non-negative")
    if beta_m <= 0:
        raise ValueError("beta_m must be positive")
    return -abs(network_distance_m - gps_displacement_m) / beta_m


def candidate_transition_log_probability(
    previous: CandidateState,
    current: CandidateState,
    gps_displacement_m: float,
    beta_m: float,
    router: RoadNetworkRouter,
) -> float:
    """Return a log transition score, or ``-inf`` for a disconnected pair."""
    network_distance = router.candidate_distance_m(previous, current)
    if network_distance is None:
        return -math.inf
    return transition_log_probability(network_distance, gps_displacement_m, beta_m)


def logsumexp(values: Sequence[float]) -> float:
    """Stable log(sum(exp(values))); an all-impossible set returns ``-inf``."""
    if not values:
        return -math.inf
    if any(isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) or value == math.inf for value in values):
        raise ValueError("log values must not contain NaN or positive infinity")
    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def normalize_log_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite log weights; preserve an all-impossible vector."""
    if any(isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) or value == math.inf for value in values):
        raise ValueError("log values must not contain NaN or positive infinity")
    normalizer = logsumexp(values)
    if normalizer == -math.inf:
        return tuple(-math.inf for _ in values)
    return tuple(value - normalizer for value in values)


def initial_log_probabilities(
    candidates: Sequence[CandidateState], priors: Sequence[float] | None = None
) -> tuple[float, ...]:
    """Return normalized initial state log probabilities.

    With no supplied priors, candidate states receive a uniform prior. Optional
    priors must be non-negative weights aligned with the candidate sequence.
    """
    if not candidates:
        return ()
    if priors is None:
        uniform_log_prior = -math.log(len(candidates))
        return tuple(uniform_log_prior for _ in candidates)
    if len(priors) != len(candidates):
        raise ValueError("priors must have one entry per candidate")
    if any(not math.isfinite(prior) or prior < 0.0 for prior in priors):
        raise ValueError("initial prior weights must be finite and non-negative")
    log_weights = tuple(-math.inf if prior == 0.0 else math.log(prior) for prior in priors)
    if all(weight == -math.inf for weight in log_weights):
        raise ValueError("at least one initial prior weight must be positive")
    return normalize_log_probabilities(log_weights)
