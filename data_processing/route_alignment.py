
import numpy as np
import pandas as pd

from shapely import Point, distance
from pyproj import Transformer


# Exploratory penalty for advancing through reference-route indices.
# We will validate and tune this value using multiple trips.
ADVANCE_PENALTY_METERS = 2.0


def monotonic_route_alignment(distance_matrix, advance_penalty):
    """
    Dynamic programming alignment.

    Route indices may stay the same or move forward.
    Backward movements are not allowed.

    Complexity: O(number of observations * number of route arcs).
    """

    number_of_observations, number_of_route_arcs = distance_matrix.shape

    if number_of_route_arcs == 0:
        raise ValueError("Reference route is empty.")

    if not np.isfinite(distance_matrix).all():
        raise ValueError("Distance matrix contains invalid values.")

    route_indices = np.arange(number_of_route_arcs)

    costs = np.full(
        (number_of_observations, number_of_route_arcs),
        np.inf
    )

    backpointers = np.full(
        (number_of_observations, number_of_route_arcs),
        -1,
        dtype=int
    )

    # First observation: prefer the beginning of the reference route,
    # but allow a later starting index if its distance is much better.
    costs[0] = (
        distance_matrix[0]
        + advance_penalty * route_indices
    )

    # Process the remaining GPS observations.
    for observation in range(1, number_of_observations):

        previous_costs = costs[observation - 1]

        # Advancing from k to j costs penalty * (j - k).
        adjusted_costs = (
            previous_costs - advance_penalty * route_indices
        )

        # Find the best previous route index among all k <= j.
        prefix_best_cost = np.empty(number_of_route_arcs)
        prefix_best_index = np.empty(number_of_route_arcs, dtype=int)

        best_cost = np.inf
        best_index = 0

        for route_index in range(number_of_route_arcs):

            if adjusted_costs[route_index] < best_cost:
                best_cost = adjusted_costs[route_index]
                best_index = route_index

            prefix_best_cost[route_index] = best_cost
            prefix_best_index[route_index] = best_index

        costs[observation] = (
            distance_matrix[observation]
            + prefix_best_cost
            + advance_penalty * route_indices
        )

        backpointers[observation] = prefix_best_index

    # Recover the lowest-cost route-index sequence.
    aligned_indices = np.empty(number_of_observations, dtype=int)

    aligned_indices[-1] = int(np.argmin(costs[-1]))

    for observation in range(number_of_observations - 1, 0, -1):
        aligned_indices[observation - 1] = backpointers[
            observation,
            aligned_indices[observation]
        ]

    return aligned_indices


def align_reference_route(
    gps,
    road_network,
    route,
    candidates
):
    """
    Align GPS observations to the ordered reference route.

    Uses dynamic programming to prevent backward route-index
    movements.
    """

    epsg_code = road_network.attrs.get("epsg_code")

    if epsg_code is None:
        raise ValueError(
            "Road network projection information is missing."
        )

    if road_network.empty:
        raise ValueError("Road network is empty.")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True
    )

    print(f"Route alignment using EPSG:{epsg_code}")

    route_ids = np.asarray(route, dtype=int)

    if len(route_ids) == 0:
        raise ValueError("Reference route is empty.")

    roads_by_id = road_network.set_index("arc_id")

    missing_route_ids = [
        arc_id for arc_id in route_ids
        if arc_id not in roads_by_id.index
    ]

    if missing_route_ids:
        raise ValueError(
            "Reference route contains unavailable road arcs: "
            f"{missing_route_ids[:10]}"
        )

    # Preserve the original reference-route order.
    route_geometries = roads_by_id.loc[
        route_ids, "geometry"
    ].to_numpy()

    # Project GPS observations into the road network's CRS.
    gps_x, gps_y = transformer.transform(
        gps["longitude"].to_numpy(),
        gps["latitude"].to_numpy()
    )

    # Calculate distances from every observation to every
    # reference-route arc.
    distance_matrix = np.empty(
        (len(gps), len(route_geometries)),
        dtype=float
    )

    for observation_id, (x, y) in enumerate(zip(gps_x, gps_y)):
        gps_point = Point(x, y)

        distance_matrix[observation_id] = distance(
            route_geometries,
            gps_point
        )

    # Find the best nondecreasing route-index sequence.
    aligned_indices = monotonic_route_alignment(
        distance_matrix,
        ADVANCE_PENALTY_METERS
    )

    aligned_arc_ids = route_ids[aligned_indices]

    aligned_distances = distance_matrix[
        np.arange(len(gps)),
        aligned_indices
    ]

    alignment = pd.DataFrame({
        "observation_id": np.arange(len(gps)),
        "route_index": aligned_indices,
        "reference_arc_id": aligned_arc_ids,
        "reference_distance_m": aligned_distances
    })

    # Check whether each selected reference arc is among
    # the generated candidates for that observation.
    candidate_sets = (
        candidates.groupby("observation_id")["arc_id"]
        .apply(set)
        .to_dict()
    )

    alignment["reference_arc_is_candidate"] = [
        int(
            row.reference_arc_id
            in candidate_sets.get(row.observation_id, set())
        )
        for row in alignment.itertuples()
    ]

    backward_movements = int(
        (alignment["route_index"].diff() < 0).sum()
    )

    matched_count = int(
        alignment["reference_arc_is_candidate"].sum()
    )

    print("Monotonic route alignment completed.")

    print("\nAlignment summary:")
    print("GPS observations:", len(alignment))
    print("Reference route arcs:", len(route_ids))
    print("Reference arcs found among candidates:", matched_count)

    print(
        "Candidate coverage:",
        f"{matched_count / len(alignment) * 100:.2f}%"
    )

    print(
        "Average distance to aligned reference arc:",
        f"{alignment['reference_distance_m'].mean():.2f} m"
    )

    print(
        "Maximum distance to aligned reference arc:",
        f"{alignment['reference_distance_m'].max():.2f} m"
    )

    print("Backward route-index movements:", backward_movements)

    print(
        "Route index advances:",
        int((alignment["route_index"].diff() > 0).sum())
    )

    print("\nFirst ten alignment records:")
    print(alignment.head(10).to_string(index=False))

    return alignment


if __name__ == "__main__":

    import sys
    from pathlib import Path

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent)
    )

    from data_loader import (
        TRAIN_DIR,
        discover_trip_directories,
        load_trip
    )

    from preprocessing import clean_gps_data
    from road_network import build_road_network
    from candidate_generation import generate_candidates

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

    alignment = align_reference_route(
        cleaned_gps,
        road_network,
        trip["route"],
        candidates
    )
