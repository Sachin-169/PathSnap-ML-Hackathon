"""Directed road-network distance utilities for candidate transitions."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict

from .types import CandidateState, RoadNetwork

_EARTH_RADIUS_M = 6_371_008.8


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between longitude/latitude coordinates."""
    values = (lon1, lat1, lon2, lat2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coordinates must be finite")
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("longitude must be in [-180, 180]")
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("latitude must be in [-90, 90]")
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = math.radians(lon2 - lon1)
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))


class RoadNetworkRouter:
    """Compute shortest directed distances between candidate positions.

    Arc lengths supplied by the data-processing module are preferred. If absent,
    the geodesic distance between arc endpoints is used as a straight-chord
    approximation. A small bounded pair cache avoids repeated Dijkstra searches.
    """

    def __init__(self, network: RoadNetwork, cache_size: int = 8192) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.network = network
        self.cache_size = cache_size
        self.arc_lengths_m: dict[int, float] = {}
        self.adjacency: dict[int, list[tuple[int, float]]] = {
            node_id: [] for node_id in network.nodes
        }
        for arc_id, arc in network.arcs.items():
            if arc.from_node_id not in network.nodes or arc.to_node_id not in network.nodes:
                raise ValueError(f"arc {arc_id} refers to a missing node")
            length = arc.length_m
            if length is None:
                start = network.nodes[arc.from_node_id]
                end = network.nodes[arc.to_node_id]
                length = haversine_distance_m(
                    start.longitude, start.latitude, end.longitude, end.latitude
                )
            if not math.isfinite(length) or length < 0:
                raise ValueError(f"arc {arc_id} length must be finite and non-negative")
            self.arc_lengths_m[arc_id] = length
            self.adjacency[arc.from_node_id].append((arc.to_node_id, length))
        self._pair_cache: OrderedDict[tuple[int, int], float | None] = OrderedDict()

    def _node_distance_m(self, source: int, target: int) -> float | None:
        key = (source, target)
        if key in self._pair_cache:
            self._pair_cache.move_to_end(key)
            return self._pair_cache[key]
        if source not in self.adjacency or target not in self.adjacency:
            raise ValueError("shortest-path endpoint is absent from the road network")
        if source == target:
            return 0.0

        distances = {source: 0.0}
        frontier = [(0.0, source)]
        result = None
        while frontier:
            distance, node = heapq.heappop(frontier)
            if distance != distances.get(node):
                continue
            if node == target:
                result = distance
                break
            for neighbour, edge_length in self.adjacency[node]:
                next_distance = distance + edge_length
                if next_distance < distances.get(neighbour, math.inf):
                    distances[neighbour] = next_distance
                    heapq.heappush(frontier, (next_distance, neighbour))
        if self.cache_size:
            self._pair_cache[key] = result
            if len(self._pair_cache) > self.cache_size:
                self._pair_cache.popitem(last=False)
        return result

    def candidate_distance_m(
        self, previous: CandidateState, current: CandidateState
    ) -> float | None:
        """Shortest directed travel distance between two candidate positions.

        Candidate fractions represent the distance along each directed arc as a
        fraction of its length. Missing fractions use the midpoint estimate.
        ``None`` means there is no directed path between the candidates.
        """
        if previous.arc_id not in self.network.arcs or current.arc_id not in self.network.arcs:
            raise ValueError("candidate arc ID is absent from the road network")
        previous_arc = self.network.arcs[previous.arc_id]
        current_arc = self.network.arcs[current.arc_id]
        previous_fraction = 0.5 if previous.fraction_along is None else previous.fraction_along
        current_fraction = 0.5 if current.fraction_along is None else current.fraction_along
        for name, fraction in (("previous", previous_fraction), ("current", current_fraction)):
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{name} candidate fraction must be finite and in [0, 1]")

        previous_length = self.arc_lengths_m[previous.arc_id]
        current_length = self.arc_lengths_m[current.arc_id]
        if (
            previous.arc_id == current.arc_id
            and current_fraction >= previous_fraction
        ):
            return (current_fraction - previous_fraction) * previous_length

        between = self._node_distance_m(previous_arc.to_node_id, current_arc.from_node_id)
        if between is None:
            return None
        return (
            (1.0 - previous_fraction) * previous_length
            + between
            + current_fraction * current_length
        )
