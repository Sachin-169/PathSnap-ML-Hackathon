"""Synthetic checks for model features at unlabeled inference time."""

import unittest

import pandas as pd
from pyproj import Transformer

from data_processing.inference_features import FEATURE_COLUMNS, create_inference_features


class InferenceFeatureTests(unittest.TestCase):
    def test_feature_columns_are_finite_and_candidate_order_is_preserved(self):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
        gx, gy = transformer.transform([-93.0, -92.9999], [44.0, 44.0])
        gps = pd.DataFrame(
            {
                "longitude": [-93.0, -92.9999],
                "latitude": [44.0, 44.0],
                "speed_mps": [0.0, 8.0],
                "time_difference_s": [0.0, 1.0],
            }
        )
        candidates = pd.DataFrame(
            [
                {"observation_id": 1, "arc_id": 1, "distance_m": 3.0,
                 "projection_fraction": 0.7, "gps_x": gx[1], "gps_y": gy[1],
                 "road_length_m": 120.0},
                {"observation_id": 1, "arc_id": 0, "distance_m": 2.0,
                 "projection_fraction": 0.2, "gps_x": gx[1], "gps_y": gy[1],
                 "road_length_m": 100.0},
            ]
        )
        network = pd.DataFrame(
            [
                {"arc_id": 0, "start_x": gx[0], "start_y": gy[0],
                 "end_x": gx[1], "end_y": gy[1]},
                {"arc_id": 1, "start_x": gx[0], "start_y": gy[0] + 120.0,
                 "end_x": gx[1], "end_y": gy[1] + 120.0},
            ]
        )
        network.attrs["epsg_code"] = 32615

        features = create_inference_features(gps, candidates, network)

        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features["distance_m"].tolist(), [3.0, 2.0])
        self.assertEqual(features["projection_fraction"].tolist(), [0.7, 0.2])
        self.assertFalse(features.isna().any().any())
        self.assertEqual(len(features), len(candidates))

    def test_empty_candidates_return_empty_feature_frame(self):
        features = create_inference_features(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertTrue(features.empty)


if __name__ == "__main__":
    unittest.main()
