"""Build a local, label-free registry from the supplied student-test zip.

This prepares the GPS, road graph, and candidate data consumed by api/main.py.
It deliberately never reads the .route files, so reference labels are not
passed to the matcher or exposed by the API.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_processing"))

from candidate_generation import generate_candidates  # noqa: E402
from inference_features import (  # noqa: E402
    FEATURE_COLUMNS,
    create_inference_features,
)
from preprocessing import clean_gps_data  # noqa: E402
from road_network import build_road_network  # noqa: E402


def _read_table(archive: zipfile.ZipFile, member: str, columns: list[str]) -> pd.DataFrame:
    try:
        with archive.open(member) as stream:
            frame = pd.read_csv(stream, sep=r"\s+", header=None, dtype=float)
    except KeyError as error:
        raise FileNotFoundError(f"Missing required student-test file: {member}") from error
    if frame.empty or frame.shape[1] != len(columns):
        raise ValueError(
            f"{member}: expected a non-empty table with {len(columns)} columns"
        )
    frame.columns = columns
    return frame


def _json_value(value: Any) -> int | float | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int,)):
        return int(value)
    number = float(value)
    return int(number) if number.is_integer() else number


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: _json_value(value) for column, value in row.items()}
        for row in frame[columns].to_dict(orient="records")
    ]


def build_registry(archive_path: Path, output_path: Path) -> None:
    prefix = "Student folder/student_test/"
    trips: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        trip_ids = sorted(
            Path(name).stem
            for name in members
            if name.startswith(prefix) and name.endswith(".track")
        )
        if not trip_ids:
            raise ValueError(f"No student_test .track files found in {archive_path}")

        for trip_id in trip_ids:
            print(f"Preparing student-test trip {trip_id}...", flush=True)
            base = f"{prefix}{trip_id}/{trip_id}"
            track = _read_table(
                archive, f"{base}.track", ["longitude", "latitude", "timestamp"]
            )
            nodes = _read_table(
                archive, f"{base}.nodes", ["longitude", "latitude"]
            )
            arcs = _read_table(
                archive, f"{base}.arcs", ["start_node", "end_node"]
            ).astype({"start_node": "int64", "end_node": "int64"})

            gps = clean_gps_data(track)
            projected_nodes, road_network = build_road_network(nodes, arcs)

            if road_network.empty:
                candidate_records: list[dict[str, Any]] = []
                feature_records: list[dict[str, Any]] = []
            else:
                try:
                    candidate_table = generate_candidates(gps, road_network)
                    candidate_records = _records(
                        candidate_table,
                        [
                            "observation_id",
                            "arc_id",
                            "distance_m",
                            "projection_fraction",
                        ],
                    )
                    feature_table = create_inference_features(
                        gps, candidate_table, road_network
                    )
                    feature_records = _records(feature_table, FEATURE_COLUMNS)
                except RuntimeError as error:
                    if "No road candidates found" not in str(error):
                        raise
                    candidate_records = []
                    feature_records = []

            gps_records = _records(
                gps, ["longitude", "latitude", "timestamp"]
            )
            node_table = projected_nodes[["longitude", "latitude"]].copy()
            node_table.insert(0, "node_id", projected_nodes.index.astype(int))
            node_records = _records(node_table, ["node_id", "longitude", "latitude"])
            arc_records = _records(
                road_network,
                ["arc_id", "start_node", "end_node", "length_m"],
            )

            trips.append(
                {
                    "trip_id": trip_id,
                    "gps": gps_records,
                    "projected_nodes": node_records,
                    "road_network": arc_records,
                    "candidates": candidate_records,
                    "candidate_features": feature_records,
                }
            )
            print(
                f"  GPS rows: {len(gps_records)}, candidates: {len(candidate_records)}",
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump({"trips": trips}, stream, separators=(",", ":"))
    print(f"Wrote {len(trips)} label-free trips to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local route-matching registry from student-test files."
    )
    parser.add_argument(
        "--student-zip",
        type=Path,
        default=ROOT / "hackathon_materials" / "Student folder.zip",
        help="Zip containing Student folder/student_test/<trip>/<trip>.* files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "demo_trips.json",
        help="Local JSON registry path (ignored by Git)",
    )
    args = parser.parse_args()
    build_registry(args.student_zip.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
