"""PathSnap HMM and Viterbi route reconstruction."""

from .emissions import emission_probability, gaussian_emission_probability
from .adapters import build_map_matching_input
from .matcher import match_trip
from .network import RoadNetworkRouter, haversine_distance_m
from .probabilities import (
    candidate_transition_log_probability,
    emission_log_probability,
    initial_log_probabilities,
    log_gaussian_emission_probability,
    log_probability,
    logsumexp,
    normalize_log_probabilities,
    transition_log_probability,
)
from .reconstruction import collapse_consecutive_arc_ids, reconstruct_arc_ids
from .transitions import candidate_transition_probability, transition_probability
from .types import (
    CandidateState,
    GPSObservation,
    MapMatchingConfig,
    MapMatchingInput,
    MapMatchingResult,
    RoadArc,
    RoadNetwork,
    RoadNode,
)
from .viterbi import DecodedPath, ViterbiLattice, viterbi_backtrack, viterbi_forward

__all__ = [
    "CandidateState",
    "DecodedPath",
    "GPSObservation",
    "MapMatchingConfig",
    "MapMatchingInput",
    "MapMatchingResult",
    "RoadArc",
    "RoadNetwork",
    "RoadNetworkRouter",
    "RoadNode",
    "ViterbiLattice",
    "candidate_transition_log_probability",
    "candidate_transition_probability",
    "build_map_matching_input",
    "collapse_consecutive_arc_ids",
    "emission_log_probability",
    "emission_probability",
    "gaussian_emission_probability",
    "haversine_distance_m",
    "initial_log_probabilities",
    "log_gaussian_emission_probability",
    "log_probability",
    "logsumexp",
    "match_trip",
    "normalize_log_probabilities",
    "reconstruct_arc_ids",
    "transition_log_probability",
    "transition_probability",
    "viterbi_backtrack",
    "viterbi_forward",
]
