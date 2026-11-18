"""Synthetic unit tests for PathSnap's HMM and Viterbi module."""

import math
import unittest

from route_reconstruction import (
    CandidateState,
    DecodedPath,
    GPSObservation,
    MapMatchingConfig,
    MapMatchingInput,
    RoadArc,
    RoadNetwork,
    RoadNode,
    emission_probability,
    initial_log_probabilities,
    log_gaussian_emission_probability,
    match_trip,
    reconstruct_arc_ids,
    viterbi_backtrack,
    viterbi_forward,
)


def connected_network() -> RoadNetwork:
    nodes = {
        0: RoadNode(0, 0.0, 0.0),
        1: RoadNode(1, 0.0001, 0.0),
        2: RoadNode(2, 0.0002, 0.0),
        3: RoadNode(3, 1.0, 1.0),
        4: RoadNode(4, 1.0001, 1.0),
    }
    arcs = {
        0: RoadArc(0, 0, 1, 11.2),
        1: RoadArc(1, 1, 2, 11.2),
        2: RoadArc(2, 3, 4, 11.2),
    }
    return RoadNetwork(nodes, arcs)


class ProbabilityTests(unittest.TestCase):
    def test_gaussian_emission_decreases_with_distance(self) -> None:
        near = emission_probability(CandidateState(0, 5.0), sigma_m=10.0)
        far = emission_probability(CandidateState(0, 20.0), sigma_m=10.0)
        self.assertGreater(near, far)
        self.assertTrue(math.isfinite(log_gaussian_emission_probability(1e6, 10.0)))

    def test_initial_probabilities_are_normalized(self) -> None:
        candidates = [CandidateState(0, 0.0), CandidateState(1, 0.0)]
        logs = initial_log_probabilities(candidates, [1.0, 3.0])
        self.assertAlmostEqual(sum(math.exp(value) for value in logs), 1.0)
        self.assertAlmostEqual(math.exp(logs[0]), 0.25)


class ViterbiTests(unittest.TestCase):
    def test_known_sequence_and_backtracking(self) -> None:
        log = math.log
        lattice = viterbi_forward(
            [log(0.6), log(0.4)],
            [[log(0.9), log(0.2)], [log(0.7), log(0.8)]],
            [[[log(0.9), log(0.1)], [log(0.1), log(0.9)]]],
        )
        decoded = viterbi_backtrack(lattice)
        self.assertEqual(decoded.state_indices, (0, 0))
        self.assertAlmostEqual(decoded.log_probability, log(0.3402))

    def test_impossible_path_is_not_fabricated(self) -> None:
        lattice = viterbi_forward([0.0], [[0.0], [0.0]], [[[-math.inf]]])
        decoded = viterbi_backtrack(lattice)
        self.assertEqual(decoded.state_indices, ())
        self.assertEqual(decoded.log_probability, -math.inf)

    def test_candidate_states_reconstruct_observation_and_route_ids(self) -> None:
        rows = [
            [CandidateState(7, 1.0), CandidateState(8, 2.0)],
            [CandidateState(7, 2.0), CandidateState(9, 1.0)],
        ]
        self.assertEqual(reconstruct_arc_ids(rows, DecodedPath((0, 0), -1.0)), ((7, 7), (7,)))


class TripMatcherTests(unittest.TestCase):
    def test_connected_road_sequence_matches(self) -> None:
        observations = [
            GPSObservation(1.0, 0.00005, 0.0),
            GPSObservation(2.0, 0.00015, 0.0),
        ]
        candidates = [[CandidateState(0, 1.0, 0.5)], [CandidateState(1, 1.0, 0.5)]]
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()))
        self.assertEqual(result.matched_arc_ids, (0, 1))
        self.assertEqual(result.route_arc_ids, (0, 1))
        self.assertEqual(result.segment_ids, (0, 0))

    def test_nearer_candidate_wins_for_single_observation(self) -> None:
        observations = [GPSObservation(1.0, 0.00005, 0.0)]
        candidates = [[CandidateState(0, 25.0, 0.5), CandidateState(1, 1.0, 0.5)]]
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()))
        self.assertEqual(result.matched_arc_ids, (1,))

    def test_disconnected_roads_start_new_segments(self) -> None:
        observations = [GPSObservation(1.0, 0.0, 0.0), GPSObservation(2.0, 1.0, 1.0)]
        candidates = [[CandidateState(0, 1.0, 0.5)], [CandidateState(2, 1.0, 0.5)]]
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()))
        self.assertEqual(result.matched_arc_ids, (0, 2))
        self.assertEqual(result.segment_ids, (0, 1))
        self.assertTrue(any("No feasible road transition" in item for item in result.diagnostics))

    def test_long_gps_gap_starts_new_segment(self) -> None:
        observations = [GPSObservation(1.0, 0.00005, 0.0), GPSObservation(30.0, 0.00015, 0.0)]
        candidates = [[CandidateState(0, 1.0, 0.5)], [CandidateState(1, 1.0, 0.5)]]
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()))
        self.assertEqual(result.segment_ids, (0, 1))
        self.assertTrue(any("GPS gap" in item for item in result.diagnostics))

    def test_missing_observation_and_no_candidates_stay_unmatched(self) -> None:
        observations = [GPSObservation(1.0, None, None), GPSObservation(2.0, 0.0, 0.0)]
        candidates = [[], []]
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()))
        self.assertEqual(result.matched_arc_ids, (None, None))
        self.assertEqual(result.segment_ids, (None, None))
        self.assertEqual(len(result.diagnostics), 2)

    def test_empty_trajectory(self) -> None:
        result = match_trip(MapMatchingInput([], [], RoadNetwork({}, {})))
        self.assertEqual(result.matched_arc_ids, ())
        self.assertIsNone(result.log_probability)

    def test_unrealistic_speed_starts_new_segment(self) -> None:
        observations = [GPSObservation(1.0, 0.00005, 0.0), GPSObservation(2.0, 0.00015, 0.0)]
        candidates = [[CandidateState(0, 1.0, 0.5)], [CandidateState(1, 1.0, 0.5)]]
        result = match_trip(
            MapMatchingInput(observations, candidates, connected_network()),
            MapMatchingConfig(max_speed_mps=1.0),
        )
        self.assertEqual(result.segment_ids, (0, 1))
        self.assertTrue(any("No feasible road transition" in item for item in result.diagnostics))

    def test_invalid_candidate_reference_fails(self) -> None:
        observations = [GPSObservation(1.0, 0.0, 0.0)]
        candidates = [[CandidateState(99, 0.0)]]
        with self.assertRaisesRegex(ValueError, "absent from the network"):
            match_trip(MapMatchingInput(observations, candidates, connected_network()))

    def test_gap_threshold_is_configurable(self) -> None:
        observations = [GPSObservation(1.0, 0.00005, 0.0), GPSObservation(3.0, 0.00015, 0.0)]
        candidates = [[CandidateState(0, 1.0, 0.5)], [CandidateState(1, 1.0, 0.5)]]
        config = MapMatchingConfig(max_gap_seconds=1.0)
        result = match_trip(MapMatchingInput(observations, candidates, connected_network()), config)
        self.assertEqual(result.segment_ids, (0, 1))


if __name__ == "__main__":
    unittest.main()
