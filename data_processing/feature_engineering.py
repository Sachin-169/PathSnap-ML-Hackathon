
import numpy as np
import pandas as pd
from pyproj import Transformer


def calculate_bearing(x1, y1, x2, y2):
    """
    Calculate bearing in degrees, measured clockwise from north.
    Supports both scalar values and NumPy/Pandas arrays.
    """
    angle = np.degrees(
        np.arctan2(
            np.asarray(x2) - np.asarray(x1),
            np.asarray(y2) - np.asarray(y1)
        )
    )

    return (angle + 360) % 360


def create_feature_dataset(gps, candidates, road_network, alignment):
    """
    Combine GPS observations, road candidates and route alignment
    into a labelled machine learning dataset.
    """

    features = candidates.copy()

    # 1. Attach reference route information.
    features = features.merge(
        alignment[
            [
                "observation_id",
                "reference_arc_id",
                "reference_distance_m"
            ]
        ],
        on="observation_id",
        how="left"
    )

    # 2. Generate binary labels.
    features["label"] = (
        features["arc_id"] == features["reference_arc_id"]
    ).astype(int)

    # 3. Attach GPS speed and time information.
    gps_features = gps[
        ["speed_mps", "time_difference_s"]
    ].copy()

    gps_features["observation_id"] = np.arange(len(gps))

    features = features.merge(
        gps_features,
        on="observation_id",
        how="left"
    )

    # 4. Attach road endpoint coordinates.
    road_details = road_network[
        [
            "arc_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y"
        ]
    ]

    features = features.merge(
        road_details,
        on="arc_id",
        how="left",
        validate="many_to_one"
    )

    # 5. Calculate road heading.
    features["road_heading"] = calculate_bearing(
        features["start_x"],
        features["start_y"],
        features["end_x"],
        features["end_y"]
    )

    # 6. Project GPS observations using the same UTM zone
    # as the road network.
    epsg_code = road_network.attrs["epsg_code"]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True
    )

    gps_x, gps_y = transformer.transform(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    # 7. Calculate GPS movement heading.
    gps_heading = np.zeros(len(gps), dtype=float)

    if len(gps) > 1:
        gps_heading[1:] = calculate_bearing(
            gps_x[:-1],
            gps_y[:-1],
            gps_x[1:],
            gps_y[1:]
        )

        gps_heading[0] = gps_heading[1]

    heading_lookup = pd.DataFrame({
        "observation_id": np.arange(len(gps)),
        "gps_heading": gps_heading
    })

    features = features.merge(
        heading_lookup,
        on="observation_id",
        how="left"
    )

    # 8. Calculate smallest angular difference.
    difference = (
        features["gps_heading"] - features["road_heading"]
    ).abs() % 360

    features["heading_difference"] = np.minimum(
        difference,
        360 - difference
    )

    # 9. Calculate distances from GPS point to road endpoints.
    features["distance_to_start_m"] = np.sqrt(
        (features["gps_x"] - features["start_x"]) ** 2
        + (features["gps_y"] - features["start_y"]) ** 2
    )

    features["distance_to_end_m"] = np.sqrt(
        (features["gps_x"] - features["end_x"]) ** 2
        + (features["gps_y"] - features["end_y"]) ** 2
    )

    # 10. Select the eight ML input features.
    feature_columns = [
        "distance_m",
        "projection_fraction",
        "road_length_m",
        "speed_mps",
        "time_difference_s",
        "heading_difference",
        "distance_to_start_m",
        "distance_to_end_m"
    ]

    # 11. Replace invalid numerical values.
    features[feature_columns] = features[
        feature_columns
    ].replace([np.inf, -np.inf], np.nan)

    # 12. Report missing values before removing invalid records.
    missing_counts = features[
        feature_columns + ["label"]
    ].isna().sum()

    print("\nMissing values before cleaning:")
    print(missing_counts)

    original_count = len(features)

    features = features.dropna(
        subset=feature_columns + ["label"]
    ).copy()

    print("\nFeature engineering completed.")
    print("Original candidate records:", original_count)
    print("Valid feature records:", len(features))
    print("Removed invalid records:", original_count - len(features))
    print("Number of features:", len(feature_columns))
    print("Positive labels:", int(features["label"].sum()))
    print("Negative labels:", int((features["label"] == 0).sum()))

    print(
        "Observations with at least one positive candidate:",
        int(features.groupby("observation_id")["label"].max().sum())
    )

    print("\nFeature columns:")
    print(feature_columns)

    print("\nFirst five feature records:")
    print(
        features[
            ["observation_id", "arc_id"]
            + feature_columns
            + ["label"]
        ].head().to_string(index=False)
    )

    return features, feature_columns
