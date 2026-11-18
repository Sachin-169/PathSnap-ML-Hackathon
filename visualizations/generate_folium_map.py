"""Generate an interactive Folium map for one student-test trip.

Reference-route data, when requested, is loaded only after HMM prediction and
used only as a display/evaluation layer. It is never passed to the classifier
or route decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.inference_features import FEATURE_COLUMNS
from route_reconstruction import build_map_matching_input, match_trip


def _reference_route_layers(
    map_object: folium.Map,
    archive_path: Path,
    trip_id: str,
) -> None:
    base = f"Student folder/student_test/{trip_id}/{trip_id}"
    with zipfile.ZipFile(archive_path) as archive:
        node_frame = pd.read_csv(
            archive.open(f"{base}.nodes"), sep=r"\s+", header=None, names=["longitude", "latitude"]
        )
        arc_frame = pd.read_csv(
            archive.open(f"{base}.arcs"), sep=r"\s+", header=None, names=["start_node", "end_node"]
        )
        route_frame = pd.read_csv(
            archive.open(f"{base}.route"), sep=r"\s+", header=None, names=["arc_id"]
        )

    group = folium.FeatureGroup(name="Reference route (evaluation only)", show=False)
    for arc_id in route_frame["arc_id"].astype(int):
        if arc_id < 0 or arc_id >= len(arc_frame):
            continue
        arc = arc_frame.iloc[arc_id]
        start_id, end_id = int(arc["start_node"]), int(arc["end_node"])
        if not (0 <= start_id < len(node_frame) and 0 <= end_id < len(node_frame)):
            continue
        start, end = node_frame.iloc[start_id], node_frame.iloc[end_id]
        folium.PolyLine(
            [(float(start.latitude), float(start.longitude)), (float(end.latitude), float(end.longitude))],
            color="#159447",
            weight=4,
            opacity=0.88,
            tooltip=f"Reference arc {arc_id}",
        ).add_to(group)
    group.add_to(map_object)


def generate_map(
    trip_id: str,
    registry_path: Path,
    model_path: Path,
    output_path: Path,
    reference_zip: Path | None = None,
) -> Path:
    with registry_path.open(encoding="utf-8") as stream:
        trips = json.load(stream)["trips"]
    record = next((item for item in trips if str(item["trip_id"]) == str(trip_id)), None)
    if record is None:
        raise KeyError(f"Trip {trip_id} is not present in {registry_path}")

    candidates = record["candidates"]
    feature_rows = record["candidate_features"]
    if len(candidates) != len(feature_rows):
        raise ValueError("Candidate features must align one-to-one with candidate rows")
    if candidates:
        model = joblib.load(model_path)
        probabilities = model.predict_proba(
            pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        )[:, list(model.classes_).index(1)].tolist()
    else:
        probabilities = []

    match_input = build_map_matching_input(
        gps=record["gps"],
        projected_nodes=record["projected_nodes"],
        road_network=record["road_network"],
        candidates=candidates,
        classifier_probabilities=probabilities,
    )
    result = match_trip(match_input)
    gps_points = [
        (observation.latitude, observation.longitude)
        for observation in match_input.observations
        if observation.latitude is not None and observation.longitude is not None
    ]
    if not gps_points:
        raise ValueError(f"Trip {trip_id} has no valid GPS coordinates")

    center = [
        float(np.mean([point[0] for point in gps_points])),
        float(np.mean([point[1] for point in gps_points])),
    ]
    map_object = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    network_layer = folium.FeatureGroup(
        name=f"Road network ({len(match_input.network.arcs)} arcs)", show=True
    )
    for arc in match_input.network.arcs.values():
        start = match_input.network.nodes[arc.from_node_id]
        end = match_input.network.nodes[arc.to_node_id]
        folium.PolyLine(
            [(start.latitude, start.longitude), (end.latitude, end.longitude)],
            color="#718096",
            weight=1,
            opacity=0.4,
        ).add_to(network_layer)
    network_layer.add_to(map_object)

    gps_layer = folium.FeatureGroup(
        name=f"GPS observations ({len(gps_points)})", show=True
    )
    for observation_id, observation in enumerate(match_input.observations):
        if observation.latitude is None or observation.longitude is None:
            continue
        folium.CircleMarker(
            location=(observation.latitude, observation.longitude),
            radius=2.5,
            color="#1769d2",
            weight=1,
            fill=True,
            fill_color="#1769d2",
            fill_opacity=0.75,
            tooltip=f"GPS observation {observation_id}",
        ).add_to(gps_layer)
    gps_layer.add_to(map_object)

    route_layer = folium.FeatureGroup(
        name=f"HMM + Viterbi route ({len(result.route_arc_ids)} arcs)", show=True
    )
    for arc_id in result.route_arc_ids:
        arc = match_input.network.arcs[arc_id]
        start = match_input.network.nodes[arc.from_node_id]
        end = match_input.network.nodes[arc.to_node_id]
        folium.PolyLine(
            [(start.latitude, start.longitude), (end.latitude, end.longitude)],
            color="#e53935",
            weight=4,
            opacity=0.95,
            tooltip=f"Predicted arc {arc_id}",
        ).add_to(route_layer)
    route_layer.add_to(map_object)

    if reference_zip is not None:
        _reference_route_layers(map_object, reference_zip, trip_id)

    if gps_points:
        folium.Marker(gps_points[0], tooltip="Trip start", icon=folium.Icon(color="green")).add_to(map_object)
        folium.Marker(gps_points[-1], tooltip="Trip end", icon=folium.Icon(color="purple")).add_to(map_object)
        map_object.fit_bounds(gps_points)
    folium.LayerControl(collapsed=False).add_to(map_object)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_id", help="Trip ID in artifacts/demo_trips.json")
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "artifacts" / "demo_trips.json"
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts" / "final_classifier.joblib"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML; defaults to artifacts/maps/trip_<trip_id>.html",
    )
    parser.add_argument(
        "--reference-zip", type=Path, default=None,
        help="Optional supplied student-test ZIP; adds its reference route as a display-only layer",
    )
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts" / "maps" / f"trip_{args.trip_id}.html"
    saved = generate_map(
        args.trip_id,
        args.registry.expanduser(),
        args.model.expanduser(),
        output.expanduser(),
        args.reference_zip.expanduser() if args.reference_zip else None,
    )
    print(f"Saved Folium map: {saved}")


if __name__ == "__main__":
    main()
