"""Observation likelihoods for road-candidate states."""

from __future__ import annotations

import math
from numbers import Real

from .types import CandidateState


def gaussian_emission_probability(distance_m: float, sigma_m: float) -> float:
    """Return a Gaussian GPS-distance likelihood density.

    The perpendicular observation-to-candidate distance is modeled as a
    zero-mean Gaussian error with scale ``sigma_m``. ``sigma_m`` should be
    estimated or selected using permitted development trips; it is configurable
    because GPS accuracy depends on the data source and environment.
    """
    if isinstance(distance_m, bool) or not isinstance(distance_m, Real) or not math.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be a finite non-negative value")
    if isinstance(sigma_m, bool) or not isinstance(sigma_m, Real) or not math.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("sigma_m must be a finite positive value")
    standardized_distance = distance_m / sigma_m
    return math.exp(-0.5 * standardized_distance * standardized_distance) / (
        math.sqrt(2.0 * math.pi) * sigma_m
    )


def emission_probability(candidate: CandidateState, sigma_m: float) -> float:
    """Return the candidate's observation likelihood.

    A supplied classifier probability is used as-is. When the candidate has no
    classifier score, use the Gaussian likelihood from its geometric distance.
    Classifier values must be probabilities, not uncalibrated decision scores.
    """
    classifier_probability = candidate.classifier_probability
    if classifier_probability is None:
        return gaussian_emission_probability(candidate.distance_m, sigma_m)
    if (
        isinstance(classifier_probability, bool)
        or not isinstance(classifier_probability, Real)
        or not math.isfinite(classifier_probability)
        or classifier_probability < 0.0
        or classifier_probability > 1.0
    ):
        raise ValueError("classifier_probability must be finite and in [0, 1]")
    return classifier_probability
