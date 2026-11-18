"""FastAPI bridge between the evaluator UI and PathSnap route reconstruction.

The API loads preprocessed, label-free trip inputs from PATHSNAP_TRIPS_JSON.
Each trip record uses the schema described in api/README.md. The classifier
probabilities field is optional until Member 2 supplies a saved inference model.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_processing.inference_features import FEATURE_COLUMNS
from route_reconstruction import build_map_matching_input, match_trip


ROOT = Path(__file__).resolve().parents[1]
TRIPS_PATH = Path(
    os.environ.get("PATHSNAP_TRIPS_JSON", ROOT / "artifacts" / "demo_trips.json")
).expanduser()
MODEL_PATH = Path(
    os.environ.get("PATHSNAP_MODEL_PATH", ROOT / "artifacts" / "final_classifier.joblib")
).expanduser()

app = FastAPI(title="PathSnap Route Matching API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _load_trips() -> dict[str, dict[str, Any]]:
    if not TRIPS_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Trip registry is not configured. Set PATHSNAP_TRIPS_JSON to a "
                "JSON file containing preprocessed, label-free trip inputs."
            ),
        )
    try:
        with TRIPS_PATH.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Could not load trip registry") from error

    trips = payload.get("trips") if isinstance(payload, dict) else None
    if not isinstance(trips, list):
        raise HTTPException(
            status_code=500, detail='Trip registry must contain a "trips" list'
        )

    indexed: dict[str, dict[str, Any]] = {}
    for record in trips:
        if not isinstance(record, dict) or "trip_id" not in record:
            raise HTTPException(
                status_code=500, detail="Every trip record must contain trip_id"
            )
        trip_id = str(record["trip_id"])
        if trip_id in indexed:
            raise HTTPException(
                status_code=500, detail=f"Duplicate trip_id in registry: {trip_id}"
            )
        indexed[trip_id] = record
    return indexed


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained classifier is missing. Run models/bayesian_tuning.py "
                "to create artifacts/final_classifier.joblib."
            ),
        )
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError) as error:
        raise HTTPException(status_code=503, detail="Could not load classifier artifact") from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "PathSnap Route Matching API"}


@app.get("/api/trips")
def list_trips() -> dict[str, list[dict[str, str]]]:
    trips = _load_trips()
    return {
        "trips": [
            {"trip_id": trip_id, "label": f"Evaluation trip {trip_id}"}
            for trip_id in trips
        ]
    }


@app.post("/api/trips/{trip_id}/match")
def match_evaluation_trip(trip_id: str) -> dict[str, Any]:
    trips = _load_trips()
    record = trips.get(trip_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id: {trip_id}")

    try:
        candidates = record["candidates"]
        candidate_features = record.get("candidate_features")
        if candidate_features is None or len(candidate_features) != len(candidates):
            raise ValueError(
                "candidate_features must align one-to-one with candidate rows"
            )
        if candidate_features:
            classifier = _load_model()
            feature_frame = pd.DataFrame(candidate_features, columns=FEATURE_COLUMNS)
            class_labels = list(classifier.classes_)
            if 1 not in class_labels:
                raise ValueError("Classifier does not contain the positive class label 1")
            positive_column = class_labels.index(1)
            classifier_probabilities = classifier.predict_proba(feature_frame)[:, positive_column].tolist()
        else:
            classifier_probabilities = []

        match_input = build_map_matching_input(
            gps=record["gps"],
            projected_nodes=record["projected_nodes"],
            road_network=record["road_network"],
            candidates=candidates,
            classifier_probabilities=classifier_probabilities,
        )
        result = match_trip(match_input)
    except KeyError as error:
        raise HTTPException(
            status_code=422, detail=f"Trip record is missing required field {error}"
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    arc_geometry = []
    for arc_id in result.route_arc_ids:
        arc = match_input.network.arcs[arc_id]
        start = match_input.network.nodes[arc.from_node_id]
        end = match_input.network.nodes[arc.to_node_id]
        arc_geometry.append(
            {
                "arc_id": arc_id,
                # Coordinates are [latitude, longitude] for convenient plotting.
                "coordinates": [
                    [start.latitude, start.longitude],
                    [end.latitude, end.longitude],
                ],
            }
        )

    road_network_geometry = []
    for arc_id, arc in match_input.network.arcs.items():
        start = match_input.network.nodes[arc.from_node_id]
        end = match_input.network.nodes[arc.to_node_id]
        road_network_geometry.append(
            {
                "arc_id": arc_id,
                "coordinates": [
                    [start.latitude, start.longitude],
                    [end.latitude, end.longitude],
                ],
            }
        )

    return {
        "trip_id": trip_id,
        "matched_arc_ids": list(result.matched_arc_ids),
        "route_arc_ids": list(result.route_arc_ids),
        "segment_ids": list(result.segment_ids),
        "road_network_geometry": road_network_geometry,
        "route_geometry": arc_geometry,
        "gps": [
            {
                "timestamp_s": observation.timestamp_s,
                "latitude": observation.latitude,
                "longitude": observation.longitude,
            }
            for observation in match_input.observations
        ],
        "log_score": result.log_score,
        "diagnostics": list(result.diagnostics),
    }
