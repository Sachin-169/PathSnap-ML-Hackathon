
import numpy as np
import pandas as pd

from shapely.geometry import LineString

from geo_utils import get_utm_transformer


def build_road_network(nodes, arcs):
    """
    Convert geographic road nodes and arcs into projected
    road segments measured in metres.

    Automatically selects the appropriate UTM zone
    for each trip.
    """

    nodes = nodes.copy()
    arcs = arcs.copy()

    # Select the projection using this trip's node coordinates.
    transformer, epsg_code = get_utm_transformer(
        nodes["longitude"],
        nodes["latitude"]
    )

    print(f"Using projection: EPSG:{epsg_code}")

    # Project geographic coordinates into metres.
    x_coordinates, y_coordinates = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )

    nodes["x"] = x_coordinates
    nodes["y"] = y_coordinates

    # Get the start and end node IDs for every arc.
    start_ids = arcs["start_node"].to_numpy(dtype=int)
    end_ids = arcs["end_node"].to_numpy(dtype=int)

    # Retrieve projected coordinates.
    start_x = nodes["x"].to_numpy()[start_ids]
    start_y = nodes["y"].to_numpy()[start_ids]

    end_x = nodes["x"].to_numpy()[end_ids]
    end_y = nodes["y"].to_numpy()[end_ids]

    # Calculate road segment lengths in metres.
    lengths = np.sqrt(
        (end_x - start_x) ** 2
        + (end_y - start_y) ** 2
    )

    # Keep only finite, non-zero-length segments.
    valid_mask = (
        np.isfinite(start_x)
        & np.isfinite(start_y)
        & np.isfinite(end_x)
        & np.isfinite(end_y)
        & np.isfinite(lengths)
        & (lengths > 0)
    )

    road_network = pd.DataFrame({
        "arc_id": np.arange(len(arcs)),
        "start_node": start_ids,
        "end_node": end_ids,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "length_m": lengths
    })

    road_network["geometry"] = [
        LineString([(sx, sy), (ex, ey)])
        for sx, sy, ex, ey in zip(
            start_x,
            start_y,
            end_x,
            end_y
        )
    ]

    road_network = road_network.loc[
        valid_mask
    ].reset_index(drop=True)

    # Store the CRS so the other modules use exactly
    # the same projection for this trip.
    road_network.attrs["epsg_code"] = epsg_code

    print("Road network construction completed.")
    print("Original road arcs:", len(arcs))
    print("Valid road segments:", len(road_network))
    print(
        "Removed invalid or zero-length segments:",
        len(arcs) - len(road_network)
    )

    return nodes, road_network


if __name__ == "__main__":

    from data_loader import (
        TRAIN_DIR,
        discover_trip_directories,
        load_trip
    )

    trip_directories = discover_trip_directories(TRAIN_DIR)

    if not trip_directories:
        raise RuntimeError("No training trips found.")

    trip = load_trip(trip_directories[0])

    projected_nodes, road_network = build_road_network(
        trip["nodes"],
        trip["arcs"]
    )

    print("\nFirst five projected road nodes:")
    print(
        projected_nodes[
            ["longitude", "latitude", "x", "y"]
        ].head().to_string(index=False)
    )

    print("\nFirst five road segments:")
    print(
        road_network[
            ["arc_id", "start_node", "end_node", "length_m"]
        ].head().to_string(index=False)
    )

    print("\nRoad segment length summary:")
    print(road_network["length_m"].describe())
