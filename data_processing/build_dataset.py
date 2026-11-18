
from pathlib import Path

import pandas as pd

from data_loader import (
    TRAIN_DIR,
    discover_trip_directories,
    load_trip
)

from preprocessing import clean_gps_data
from road_network import build_road_network
from candidate_generation import generate_candidates
from route_alignment import align_reference_route
from feature_engineering import create_feature_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "candidate_features.csv"


def build_dataset():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trip_directories = discover_trip_directories(TRAIN_DIR)

    if not trip_directories:
        raise RuntimeError("No training trips found.")

    print("Total training trips found:", len(trip_directories))
    print("\nStarting dataset generation...\n")

    total_records = 0
    successful_trips = 0

    # Process one trip at a time.
    # Each trip has its own road network.
    for trip_number, trip_directory in enumerate(
        trip_directories, start=1
    ):

        print("\n" + "=" * 60)
        print(
            f"Processing trip {trip_number}/{len(trip_directories)}: "
            f"{trip_directory.name}"
        )
        print("=" * 60)

        trip = load_trip(trip_directory)

        # Step 1: Clean GPS observations.
        gps = clean_gps_data(trip["track"])

        # Step 2: Construct this trip's road network.
        projected_nodes, road_network = build_road_network(
            trip["nodes"],
            trip["arcs"]
        )

        # Step 3: Generate nearby road candidates.
        candidates = generate_candidates(
            gps,
            road_network
        )

        # Step 4: Align the reference route.
        alignment = align_reference_route(
            gps,
            road_network,
            trip["route"],
            candidates
        )

        # Step 5: Create labelled ML features.
        feature_dataset, feature_columns = create_feature_dataset(
            gps,
            candidates,
            road_network,
            alignment
        )

        # Preserve trip identity for group-based model evaluation.
        feature_dataset.insert(
            0,
            "trip_id",
            trip["trip_id"]
        )

        # Append the current trip to the combined CSV.
        write_mode = "w" if trip_number == 1 else "a"

        feature_dataset.to_csv(
            OUTPUT_FILE,
            mode=write_mode,
            header=(trip_number == 1),
            index=False
        )

        trip_records = len(feature_dataset)
        total_records += trip_records
        successful_trips += 1

        print("\nTrip completed:", trip["trip_id"])
        print("Records generated:", trip_records)
        print("Cumulative records:", total_records)

        del feature_dataset
        del candidates
        del road_network
        del projected_nodes
        del alignment

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)

    print("Successful trips:", successful_trips)
    print("Total candidate records:", total_records)
    print("Output file:", OUTPUT_FILE)

    # Read the saved CSV to verify the combined dataset.
    combined = pd.read_csv(OUTPUT_FILE)

    print("\nCombined dataset shape:", combined.shape)

    print(
        "Unique trips:",
        combined["trip_id"].nunique()
    )

    print("\nCombined label distribution:")
    print(combined["label"].value_counts())

    print("\nFirst five combined records:")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    build_dataset()
