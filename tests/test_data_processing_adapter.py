"""Synthetic tests for adapting preprocessing records to the HMM contract."""

import unittest

from route_reconstruction.adapters import build_map_matching_input
from route_reconstruction.matcher import match_trip


class PreprocessingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gps = [
            {"longitude": -93.0, "latitude": 44.0, "timestamp": 1.0},
            {"longitude": -93.0001, "latitude": 44.0, "timestamp": 2.0},
            {"longitude": -93.0002, "latitude": 44.0, "timestamp": 3.0},
        ]
        self.nodes = [
            {"longitude": -93.0, "latitude": 44.0},
            {"longitude": -93.0001, "latitude": 44.0},
            {"longitude": -93.0002, "latitude": 44.0},
        ]
        self.arcs = [
            {"arc_id": 0, "start_node": 0, "end_node": 1, "length_m": 8.0},
            {"arc_id": 1, "start_node": 1, "end_node": 2, "length_m": 8.0},
        ]

    def test_maps_actual_column_names_and_groups_candidates(self) -> None:
        candidates = [
            {
                "observation_id": 0,
                "arc_id": 0,
                "distance_m": 1.5,
                "projection_fraction": 0.25,
            },
            {
                "observation_id": 2,
                "arc_id": 1,
                "distance_m": 2.5,
                "projection_fraction": 0.75,
            },
        ]
        adapted = build_map_matching_input(
            self.gps, self.nodes, self.arcs, candidates
        )

        self.assertEqual(adapted.observations[1].timestamp_s, 2.0)
        self.assertEqual(adapted.network.arcs[0].from_node_id, 0)
        self.assertEqual(adapted.network.arcs[1].to_node_id, 2)
        self.assertEqual(adapted.candidates_by_observation[0][0].fraction_along, 0.25)
        self.assertEqual(adapted.candidates_by_observation[1], [])
        self.assertEqual(adapted.candidates_by_observation[2][0].arc_id, 1)

    def test_classifier_probabilities_follow_candidate_row_order(self) -> None:
        candidates = [
            {"observation_id": 0, "arc_id": 0, "distance_m": 1.0, "projection_fraction": 0.2},
            {"observation_id": 0, "arc_id": 1, "distance_m": 2.0, "projection_fraction": 0.8},
        ]
        adapted = build_map_matching_input(
            self.gps, self.nodes, self.arcs, candidates,
            classifier_probabilities=[0.8, 0.2],
        )
        self.assertEqual(
            [state.classifier_probability for state in adapted.candidates_by_observation[0]],
            [0.8, 0.2],
        )

    def test_adapted_trip_runs_through_hmm_and_viterbi(self) -> None:
        candidates = [
            {"observation_id": 0, "arc_id": 0, "distance_m": 1.0, "projection_fraction": 0.1},
            {"observation_id": 1, "arc_id": 0, "distance_m": 1.0, "projection_fraction": 0.8},
            {"observation_id": 2, "arc_id": 1, "distance_m": 1.0, "projection_fraction": 0.2},
        ]
        adapted = build_map_matching_input(
            self.gps, self.nodes, self.arcs, candidates
        )
        result = match_trip(adapted)

        self.assertEqual(list(result.matched_arc_ids), [0, 0, 1])
        self.assertEqual(list(result.route_arc_ids), [0, 1])

    def test_dataframe_style_node_index_is_preserved(self) -> None:
        class FrameLike:
            def iterrows(self):
                yield 10, {"longitude": -93.0, "latitude": 44.0}
                yield 11, {"longitude": -93.0001, "latitude": 44.0}

        nodes = FrameLike()
        arcs = [{"arc_id": 0, "start_node": 10, "end_node": 11, "length_m": 8.0}]
        adapted = build_map_matching_input(self.gps, nodes, arcs, [])
        self.assertIn(10, adapted.network.nodes)
        self.assertEqual(adapted.network.arcs[0].to_node_id, 11)

    def test_probability_length_must_match_candidate_rows(self) -> None:
        candidates = [{"observation_id": 0, "arc_id": 0, "distance_m": 1.0}]
        with self.assertRaisesRegex(ValueError, "align one-to-one"):
            build_map_matching_input(
                self.gps, self.nodes, self.arcs, candidates,
                classifier_probabilities=[0.5, 0.5],
            )

    def test_invalid_candidate_observation_id_fails(self) -> None:
        candidates = [{"observation_id": 3, "arc_id": 0, "distance_m": 1.0}]
        with self.assertRaisesRegex(ValueError, "outside the GPS rows"):
            build_map_matching_input(self.gps, self.nodes, self.arcs, candidates)


if __name__ == "__main__":
    unittest.main()
