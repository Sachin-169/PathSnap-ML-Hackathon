
import numpy as np
import pandas as pd

from shapely import STRtree, points
from pyproj import Transformer


# Maximum distance from a GPS observation to a candidate road.
CANDIDATE_RADIUS_METERS = 100


def generate_candidates(gps, road_network):
    """
    Find road segments within the candidate radius
    of every GPS observation.

    Uses the same UTM projection as road_network.py.
    """

    if road_network.empty:
        raise RuntimeError("The road network contains no valid segments.")

    epsg_code = road_network.attrs.get("epsg_code")

    if epsg_code is None:
        raise ValueError(
            "Road network projection information is missing. "
            "Build the road network using build_road_network() first."
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True
    )

    print(f"Candidate generation using EPSG:{epsg_code}")

    # Build a spatial index of road geometries.
    geometries = road_network["geometry"].to_list()
    spatial_index = STRtree(geometries)

    # Project GPS coordinates into the same UTM zone.
    gps_x, gps_y = transformer.transform(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    gps_points = points(gps_x, gps_y)

    candidate_records = []

    # Search nearby roads for every GPS observation.
    for observation_id, gps_point in enumerate(gps_points):

        nearby_indices = spatial_index.query(
            gps_point,
            predicate="dwithin",
            distance=CANDIDATE_RADIUS_METERS
        )

        for road_index in nearby_indices:

            road = road_network.iloc[road_index]

            distance = gps_point.distance(road["geometry"])

            projection_fraction = (
                road["geometry"].project(gps_point)
                / road["length_m"]
            )

            candidate_records.append({
                "observation_id": observation_id,
                "arc_id": int(road["arc_id"]),
                "distance_m": float(distance),
                "projection_fraction": float(projection_fraction),
                "gps_x": float(gps_x[observation_id]),
                "gps_y": float(gps_y[observation_id]),
                "road_length_m": float(road["length_m"])
            })

    candidates = pd.DataFrame(candidate_records)

    if candidates.empty:
        raise RuntimeError(
            "No road candidates found. Check the projection "
            "and candidate radius."
        )

    print("Candidate generation completed.")
    print("GPS observations:", len(gps))
    print("Candidate records:", len(candidates))
    print(
        "Observations with candidates:",
        candidates["observation_id"].nunique()
    )

    print(
        "Average candidates per observation:",
        round(len(candidates) / len(gps), 2)
    )

    print(
        "Maximum candidates for one observation:",
        candidates.groupby("observation_id").size().max()
    )

    return candidates


if __name__ == "__main__":

    from data_loader import (
        TRAIN_DIR,
        discover_trip_directories,
        load_trip
    )

    from preprocessing import clean_gps_data
    from road_network import build_road_network

    trip_directories = discover_trip_directories(TRAIN_DIR)

    if not trip_directories:
        raise RuntimeError("No training trips found.")

    trip = load_trip(trip_directories[0])

    cleaned_gps = clean_gps_data(trip["track"])

    projected_nodes, road_network = build_road_network(
        trip["nodes"],
        trip["arcs"]
    )

    candidates = generate_candidates(
        cleaned_gps,
        road_network
    )

    print("\nFirst ten candidate records:")
    print(candidates.head(10).to_string(index=False))

    print("\nCandidate distance summary:")
    print(candidates["distance_m"].describe())
