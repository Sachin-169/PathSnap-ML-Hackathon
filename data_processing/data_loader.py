
from pathlib import Path

import numpy as np
import pandas as pd


# Project dataset location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "student_test"


def read_numeric_file(file_path, expected_columns):
    """
    Read a whitespace-separated numeric dataset file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        dtype=float
    )

    if data.empty:
        raise ValueError(f"File is empty: {file_path}")

    if data.shape[1] != expected_columns:
        raise ValueError(
            f"{file_path.name}: Expected {expected_columns} columns, "
            f"but found {data.shape[1]}"
        )

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"Invalid numeric values found in {file_path}")

    return data


def load_trip(trip_directory):
    """
    Load all four files belonging to one trip.
    """

    trip_directory = Path(trip_directory)
    trip_id = trip_directory.name

    track = read_numeric_file(
        trip_directory / f"{trip_id}.track", 3
    )

    nodes = read_numeric_file(
        trip_directory / f"{trip_id}.nodes", 2
    )

    arcs = read_numeric_file(
        trip_directory / f"{trip_id}.arcs", 2
    )

    route = read_numeric_file(
        trip_directory / f"{trip_id}.route", 1
    )

    # Convert identifiers to integers.
    arcs = arcs.astype("int64")
    route = route.iloc[:, 0].astype("int64")

    # Give columns meaningful names.
    track.columns = ["longitude", "latitude", "timestamp"]
    nodes.columns = ["longitude", "latitude"]
    arcs.columns = ["start_node", "end_node"]

    # Check that road connections refer to valid nodes.
    if (
        (arcs["start_node"] < 0).any()
        or (arcs["end_node"] < 0).any()
        or (arcs["start_node"] >= len(nodes)).any()
        or (arcs["end_node"] >= len(nodes)).any()
    ):
        raise ValueError(f"Invalid node ID found in trip {trip_id}")

    # Check that reference route IDs refer to valid arcs.
    if (
        (route < 0).any()
        or (route >= len(arcs)).any()
    ):
        raise ValueError(f"Invalid route arc ID found in trip {trip_id}")

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }


def discover_trip_directories(dataset_directory):
    """
    Find all trip directories containing .track files.
    """

    dataset_directory = Path(dataset_directory)

    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_directory}"
        )

    trip_directories = sorted(
        path for path in dataset_directory.iterdir()
        if path.is_dir()
        and (path / f"{path.name}.track").exists()
    )

    return trip_directories


if __name__ == "__main__":

    print("Loading first training trip...")

    trip_directories = discover_trip_directories(TRAIN_DIR)

    if not trip_directories:
        raise RuntimeError("No training trip directories found.")

    trip = load_trip(trip_directories[0])

    print("\nTrip loaded successfully!")
    print("Trip ID:", trip["trip_id"])

    print("\nDataset sizes:")
    print("GPS observations:", len(trip["track"]))
    print("Road nodes:", len(trip["nodes"]))
    print("Road arcs:", len(trip["arcs"]))
    print("Reference route arcs:", len(trip["route"]))

    print("\nFirst five GPS observations:")
    print(trip["track"].head())

    print("\nFirst five road nodes:")
    print(trip["nodes"].head())

    print("\nFirst five road arcs:")
    print(trip["arcs"].head())

    print("\nFirst five reference route IDs:")
    print(trip["route"].head().tolist())
