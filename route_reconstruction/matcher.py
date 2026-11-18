"""Trip-level HMM route matching with segment recovery at gaps and breaks."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

from .network import RoadNetworkRouter, haversine_distance_m
from .probabilities import (
    emission_log_probability,
    transition_log_probability,
    initial_log_probabilities,
    normalize_log_probabilities,
)
from .reconstruction import reconstruct_arc_ids
from .types import (
    CandidateState,
    GPSObservation,
    RoadNetwork,
    MapMatchingConfig,
    MapMatchingInput,
    MapMatchingResult,
)
from .viterbi import viterbi_backtrack, viterbi_forward


def _valid_observation(observation: GPSObservation) -> bool:
    values = (observation.timestamp_s, observation.longitude, observation.latitude)
    if any(
        value is None or isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in values
    ):
        return False
    assert observation.longitude is not None and observation.latitude is not None
    return (
        -180.0 <= observation.longitude <= 180.0
        and -90.0 <= observation.latitude <= 90.0
    )


def _transition_matrix(
    previous_observation: GPSObservation,
    current_observation: GPSObservation,
    previous_candidates: Sequence[CandidateState],
    current_candidates: Sequence[CandidateState],
    router: RoadNetworkRouter,
    config: MapMatchingConfig,
) -> tuple[tuple[float, ...], ...]:
    assert previous_observation.longitude is not None
    assert previous_observation.latitude is not None
    assert current_observation.longitude is not None
    assert current_observation.latitude is not None
    assert previous_observation.timestamp_s is not None
    assert current_observation.timestamp_s is not None
    elapsed_s = current_observation.timestamp_s - previous_observation.timestamp_s
    gps_distance_m = haversine_distance_m(
        previous_observation.longitude,
        previous_observation.latitude,
        current_observation.longitude,
        current_observation.latitude,
    )
    rows: list[tuple[float, ...]] = []
    for previous in previous_candidates:
        row: list[float] = []
        for current in current_candidates:
            network_distance_m = router.candidate_distance_m(previous, current)
            if network_distance_m is None or network_distance_m / elapsed_s > config.max_speed_mps:
                row.append(-math.inf)
            else:
                row.append(
                    transition_log_probability(
                        network_distance_m, gps_distance_m, config.transition_beta_m
                    )
                )
        rows.append(normalize_log_probabilities(tuple(row)))
    return tuple(rows)


def _forward_scores(
    previous_scores: Sequence[float],
    transition_matrix: Sequence[Sequence[float]],
    emissions: Sequence[float],
) -> tuple[float, ...]:
    scores: list[float] = []
    for current_index, emission in enumerate(emissions):
        best = max(
            (previous_scores[i] + transition_matrix[i][current_index]
             for i in range(len(previous_scores))),
            default=-math.inf,
        )
        scores.append(best + emission)
    return tuple(scores)


def match_trip(
    trip: MapMatchingInput, config: MapMatchingConfig = MapMatchingConfig()
) -> MapMatchingResult:
    """Match a trip, decoding contiguous feasible observation segments.

    Missing/invalid GPS rows and rows without candidates remain ``None`` in the
    observation-aligned output. Long time gaps and infeasible transitions start
    a new independent Viterbi segment and are recorded in ``diagnostics``.
    """
    if not isinstance(trip, MapMatchingInput):
        raise TypeError("trip must be a MapMatchingInput")
    if not isinstance(config, MapMatchingConfig):
        raise TypeError("config must be a MapMatchingConfig")
    observations = trip.observations
    candidates = trip.candidates_by_observation
    if len(observations) != len(candidates):
        raise ValueError("one candidate row is required for each observation")
    if not isinstance(trip.network, RoadNetwork):
        raise TypeError("network must be a RoadNetwork")
    if not observations:
        return MapMatchingResult((), (), None, (), ())
    for index, (observation, row) in enumerate(zip(observations, candidates)):
        if not isinstance(observation, GPSObservation):
            raise TypeError(f"observation {index} must be a GPSObservation")
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"candidate row {index} must be a sequence")
        for candidate in row:
            if not isinstance(candidate, CandidateState):
                raise TypeError(f"candidate row {index} contains a non-CandidateState value")
            if candidate.arc_id not in trip.network.arcs:
                raise ValueError(f"candidate arc {candidate.arc_id} is absent from the network")
    matched: list[int | None] = [None] * len(observations)
    segment_ids: list[int | None] = [None] * len(observations)
    route: list[int] = []
    diagnostics: list[str] = []
    segment_log_scores: list[float] = []
    router = RoadNetworkRouter(trip.network)

    pending_indices: list[int] = []
    pending_transition_matrices: list[tuple[tuple[float, ...], ...]] = []
    pending_emissions: list[tuple[float, ...]] = []
    pending_initial: tuple[float, ...] = ()
    pending_scores: tuple[float, ...] = ()
    next_segment_id = 0

    def flush_pending() -> None:
        nonlocal pending_indices, pending_transition_matrices, pending_emissions
        nonlocal pending_initial, pending_scores, next_segment_id
        if not pending_indices:
            return
        lattice = viterbi_forward(
            pending_initial, pending_emissions, pending_transition_matrices
        )
        decoded = viterbi_backtrack(lattice)
        if decoded.state_indices:
            rows = [candidates[index] for index in pending_indices]
            matched_ids, segment_route = reconstruct_arc_ids(rows, decoded)
            for observation_index, arc_id in zip(pending_indices, matched_ids):
                matched[observation_index] = arc_id
                segment_ids[observation_index] = next_segment_id
            route.extend(segment_route)
            segment_log_scores.append(decoded.log_score)
            next_segment_id += 1
        else:
            diagnostics.append(
                f"No feasible state path for observations {pending_indices[0]}-"
                f"{pending_indices[-1]}; left them unmatched."
            )
        pending_indices = []
        pending_transition_matrices = []
        pending_emissions = []
        pending_initial = ()
        pending_scores = ()

    for index, (observation, row) in enumerate(zip(observations, candidates)):
        if not _valid_observation(observation):
            flush_pending()
            diagnostics.append(f"Observation {index} has invalid or missing GPS/time data.")
            continue
        if not row:
            flush_pending()
            diagnostics.append(f"Observation {index} has no candidate road states.")
            continue

        if pending_indices:
            previous_index = pending_indices[-1]
            previous_observation = observations[previous_index]
            assert previous_observation.timestamp_s is not None
            assert observation.timestamp_s is not None
            elapsed_s = observation.timestamp_s - previous_observation.timestamp_s
            if elapsed_s <= 0.0:
                flush_pending()
                diagnostics.append(
                    f"Non-increasing timestamp before observation {index}; started a new segment."
                )
            elif elapsed_s > config.max_gap_seconds:
                flush_pending()
                diagnostics.append(
                    f"GPS gap of {elapsed_s:g}s before observation {index}; started a new segment."
                )
            else:
                transition_matrix = _transition_matrix(
                    previous_observation,
                    observation,
                    candidates[previous_index],
                    row,
                    router,
                    config,
                )
                current_emissions = tuple(
                    emission_log_probability(candidate, config.emission_sigma_m)
                    for candidate in row
                )
                extended_scores = _forward_scores(
                    pending_scores, transition_matrix, current_emissions
                )
                if not any(math.isfinite(score) for score in extended_scores):
                    flush_pending()
                    diagnostics.append(
                        f"No feasible road transition to observation {index}; started a new segment."
                    )
                else:
                    pending_indices.append(index)
                    pending_transition_matrices.append(transition_matrix)
                    pending_emissions.append(current_emissions)
                    pending_scores = extended_scores
                    continue

        current_emissions = tuple(
            emission_log_probability(candidate, config.emission_sigma_m)
            for candidate in row
        )
        pending_indices = [index]
        pending_transition_matrices = []
        pending_emissions = [current_emissions]
        pending_initial = initial_log_probabilities(row)
        pending_scores = tuple(
            prior + emission for prior, emission in zip(pending_initial, current_emissions)
        )

    flush_pending()
    total_log_score = (
        math.fsum(segment_log_scores) if segment_log_scores else None
    )
    return MapMatchingResult(
        matched_arc_ids=tuple(matched),
        route_arc_ids=tuple(route),
        log_score=total_log_score,
        diagnostics=tuple(diagnostics),
        segment_ids=tuple(segment_ids),
    )
