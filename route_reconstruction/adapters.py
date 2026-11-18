"""Adapters from the current data-processing tables to the HMM input types.

This module intentionally does not import pandas or NumPy. It accepts either
pandas DataFrames (via their ``to_dict``/``iterrows`` methods) or sequences of
mapping records, keeping the HMM package usable without those dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .types import (
    CandidateState,
    GPSObservation,
    MapMatchingInput,
    RoadArc,
    RoadNetwork,
    RoadNode,
)


def _records(table: Any, name: str) -> list[Mapping[str, Any]]:
    if hasattr(table, "to_dict"):
        records = table.to_dict(orient="records")
    elif isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        records = list(table)
    else:
        raise TypeError(f"{name} must be a DataFrame or a sequence of row mappings")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError(f"every row in {name} must be a mapping")
    return records


def _records_with_index(table: Any, name: str) -> list[tuple[Any, Mapping[str, Any]]]:
    if hasattr(table, "iterrows"):
        result = []
        for index, row in table.iterrows():
            record = row.to_dict() if hasattr(row, "to_dict") else row
            if not isinstance(record, Mapping):
                raise TypeError(f"every row in {name} must be a mapping")
            result.append((index, record))
        return result
    return list(enumerate(_records(table, name)))


def _integer_id(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer ID")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer ID") from error
    if isinstance(value, Real) and (not math.isfinite(value) or value != result):
        raise ValueError(f"{name} must be an integer ID")
    if isinstance(value, str) and str(result) != value.strip().lstrip("+"):
        raise ValueError(f"{name} must be an integer ID")
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability_values(values: Any, expected_length: int) -> list[float] | None:
    if values is None:
        return None
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("classifier_probabilities must be a one-dimensional sequence")
    if len(values) != expected_length:
        raise ValueError(
            "classifier_probabilities must align one-to-one with candidate rows"
        )
    return [_finite_float(value, "classifier probability") for value in values]


def build_map_matching_input(
    gps: Any,
    projected_nodes: Any,
    road_network: Any,
    candidates: Any,
    *,
    classifier_probabilities: Any = None,
) -> MapMatchingInput:
    """Adapt current preprocessing outputs into ``MapMatchingInput``.

    Expected current columns are:

    - GPS: ``longitude``, ``latitude``, ``timestamp``.
    - Projected nodes: ``longitude``, ``latitude``; the DataFrame index (or
      sequence row position) is the node ID unless ``node_id`` is present.
    - Road arcs: ``arc_id``, ``start_node``, ``end_node``, ``length_m``.
    - Candidates: ``observation_id``, ``arc_id``, ``distance_m``, and
      ``projection_fraction``.

    The optional probability sequence must be in exactly the same row order as
    ``candidates``. Candidate rows are grouped by ``observation_id``; GPS
    observations with no candidate receive an empty candidate list. The result
    aligns to the cleaned GPS row order used by candidate generation.
    """
    gps_records = _records(gps, "gps")
    node_records = _records_with_index(projected_nodes, "projected_nodes")
    arc_records = _records(road_network, "road_network")
    candidate_records = _records(candidates, "candidates")
    probabilities = _probability_values(
        classifier_probabilities, len(candidate_records)
    )

    observations = [
        GPSObservation(
            timestamp_s=_finite_float(
                row.get("timestamp_s", row.get("timestamp")), "timestamp"
            ),
            longitude=_finite_float(row.get("longitude"), "longitude"),
            latitude=_finite_float(row.get("latitude"), "latitude"),
        )
        for row in gps_records
    ]

    nodes: dict[int, RoadNode] = {}
    for dataframe_index, row in node_records:
        node_id = _integer_id(row.get("node_id", dataframe_index), "node ID")
        if node_id in nodes:
            raise ValueError(f"duplicate road node ID {node_id}")
        nodes[node_id] = RoadNode(
            node_id=node_id,
            longitude=_finite_float(row.get("longitude"), "node longitude"),
            latitude=_finite_float(row.get("latitude"), "node latitude"),
        )

    arcs: dict[int, RoadArc] = {}
    for row in arc_records:
        arc_id = _integer_id(row.get("arc_id"), "arc ID")
        if arc_id in arcs:
            raise ValueError(f"duplicate road arc ID {arc_id}")
        arcs[arc_id] = RoadArc(
            arc_id=arc_id,
            from_node_id=_integer_id(
                row.get("from_node_id", row.get("start_node")), "arc start node ID"
            ),
            to_node_id=_integer_id(
                row.get("to_node_id", row.get("end_node")), "arc end node ID"
            ),
            length_m=(
                None
                if row.get("length_m") is None
                else _finite_float(row["length_m"], "arc length")
            ),
        )

    candidates_by_observation: list[list[CandidateState]] = [
        [] for _ in observations
    ]
    for row_index, row in enumerate(candidate_records):
        observation_id = _integer_id(row.get("observation_id"), "observation ID")
        if observation_id >= len(observations):
            raise ValueError(
                f"candidate observation ID {observation_id} is outside the GPS rows"
            )
        candidate_probability = None if probabilities is None else probabilities[row_index]
        candidates_by_observation[observation_id].append(
            CandidateState(
                arc_id=_integer_id(row.get("arc_id"), "candidate arc ID"),
                distance_m=_finite_float(row.get("distance_m"), "candidate distance"),
                fraction_along=(
                    None
                    if row.get("fraction_along", row.get("projection_fraction")) is None
                    else _finite_float(
                        row.get("fraction_along", row.get("projection_fraction")),
                        "candidate fraction",
                    )
                ),
                classifier_probability=candidate_probability,
            )
        )

    return MapMatchingInput(
        observations=observations,
        candidates_by_observation=candidates_by_observation,
        network=RoadNetwork(nodes=nodes, arcs=arcs),
    )
