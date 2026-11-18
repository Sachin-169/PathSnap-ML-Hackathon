"""Build model features for unlabeled candidate roads at inference time.

This mirrors the eight model inputs in feature_engineering.py without reading
or deriving route labels. Output order stays aligned one-to-one with the
candidate DataFrame so classifier probabilities can be passed to the HMM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


FEATURE_COLUMNS = [
    "distance_m",
    "projection_fraction",
    "road_length_m",
    "speed_mps",
    "time_difference_s",
    "heading_difference",
    "distance_to_start_m",
    "distance_to_end_m",
]


def _bearing(x1, y1, x2, y2):
    angle = np.degrees(
        np.arctan2(np.asarray(x2) - np.asarray(x1), np.asarray(y2) - np.asarray(y1))
    )
    return (angle + 360) % 360


def create_inference_features(gps, candidates, road_network) -> pd.DataFrame:
    """Return unlabeled candidate features in the original candidate-row order.

    Args:
        gps: Cleaned per-trip GPS DataFrame from ``clean_gps_data``.
        candidates: Per-trip candidate DataFrame from ``generate_candidates``.
        road_network: Per-trip network from ``build_road_network`` with its
            ``epsg_code`` attribute intact.
    """
    if candidates.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    epsg_code = road_network.attrs.get("epsg_code")
    if epsg_code is None:
        raise ValueError("Road network projection information is missing.")

    features = candidates.copy()
    gps_features = gps[["speed_mps", "time_difference_s"]].copy()
    gps_features["observation_id"] = np.arange(len(gps))
    features = features.merge(
        gps_features,
        on="observation_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    road_details = road_network[
        ["arc_id", "start_x", "start_y", "end_x", "end_y"]
    ]
    features = features.merge(
        road_details,
        on="arc_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    features["road_heading"] = _bearing(
        features["start_x"],
        features["start_y"],
        features["end_x"],
        features["end_y"],
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True
    )
    gps_x, gps_y = transformer.transform(
        gps["longitude"].to_numpy(), gps["latitude"].to_numpy()
    )

    gps_heading = np.zeros(len(gps), dtype=float)
    if len(gps) > 1:
        gps_heading[1:] = _bearing(
            gps_x[:-1], gps_y[:-1], gps_x[1:], gps_y[1:]
        )
        gps_heading[0] = gps_heading[1]

    heading_lookup = pd.DataFrame(
        {"observation_id": np.arange(len(gps)), "gps_heading": gps_heading}
    )
    features = features.merge(
        heading_lookup,
        on="observation_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    heading_delta = (features["gps_heading"] - features["road_heading"]).abs() % 360
    features["heading_difference"] = np.minimum(heading_delta, 360 - heading_delta)
    features["distance_to_start_m"] = np.hypot(
        features["gps_x"] - features["start_x"],
        features["gps_y"] - features["start_y"],
    )
    features["distance_to_end_m"] = np.hypot(
        features["gps_x"] - features["end_x"],
        features["gps_y"] - features["end_y"],
    )

    result = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if result.isna().any().any():
        bad_columns = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Non-finite inference features in columns: {bad_columns}")
    return result.reset_index(drop=True)
