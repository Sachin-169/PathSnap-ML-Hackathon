
import numpy as np
import pandas as pd

from pyproj import Geod


# WGS84 ellipsoid for calculating geographic distances
geod = Geod(ellps="WGS84")


def clean_gps_data(track):
    """
    Clean GPS observations and calculate movement features.
    """

    # Work on a copy to preserve the original dataset.
    gps = track.copy()

    original_count = len(gps)

    # Remove rows containing missing or infinite values.
    gps = gps.replace([np.inf, -np.inf], np.nan)
    gps = gps.dropna(
        subset=["longitude", "latitude", "timestamp"]
    )

    # Keep geographically valid coordinates.
    gps = gps[
        gps["longitude"].between(-180, 180)
        & gps["latitude"].between(-90, 90)
    ]

    # Sort observations chronologically.
    gps = gps.sort_values("timestamp")

    # Remove duplicate timestamps, keeping the first observation.
    gps = gps.drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )

    gps = gps.reset_index(drop=True)

    if len(gps) < 2:
        raise ValueError(
            "At least two valid GPS observations are required."
        )

    # Calculate distance between consecutive observations.
    distances = [0.0]
    time_differences = [0.0]

    for i in range(1, len(gps)):

        lon1 = gps.loc[i - 1, "longitude"]
        lat1 = gps.loc[i - 1, "latitude"]

        lon2 = gps.loc[i, "longitude"]
        lat2 = gps.loc[i, "latitude"]

        _, _, distance = geod.inv(
            lon1, lat1,
            lon2, lat2
        )

        time_difference = (
            gps.loc[i, "timestamp"]
            - gps.loc[i - 1, "timestamp"]
        )

        distances.append(distance)
        time_differences.append(time_difference)

    gps["distance_m"] = distances
    gps["time_difference_s"] = time_differences

    # Calculate speed in metres per second.
    gps["speed_mps"] = np.where(
        gps["time_difference_s"] > 0,
        gps["distance_m"] / gps["time_difference_s"],
        0.0
    )

    print("GPS preprocessing completed.")
    print("Original observations:", original_count)
    print("Clean observations:", len(gps))
    print("Removed observations:", original_count - len(gps))

    return gps


if __name__ == "__main__":

    from data_loader import TRAIN_DIR, discover_trip_directories, load_trip

    trip_directories = discover_trip_directories(TRAIN_DIR)

    if not trip_directories:
        raise RuntimeError("No training trips found.")

    trip = load_trip(trip_directories[0])

    cleaned_gps = clean_gps_data(trip["track"])

    print("\nFirst five cleaned GPS observations:")
    print(cleaned_gps.head().to_string(index=False))

    print("\nGPS feature summary:")
    print(
        cleaned_gps[
            ["distance_m", "time_difference_s", "speed_mps"]
        ].describe()
    )
