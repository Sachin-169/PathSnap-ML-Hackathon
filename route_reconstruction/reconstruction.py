"""Convert decoded state indices into road-segment route identifiers."""

from __future__ import annotations

from typing import Sequence

from .types import CandidateState
from .viterbi import DecodedPath


def collapse_consecutive_arc_ids(matched_arc_ids: Sequence[int]) -> tuple[int, ...]:
    """Collapse repeated adjacent arc IDs while preserving route order."""
    collapsed: list[int] = []
    for arc_id in matched_arc_ids:
        if not collapsed or arc_id != collapsed[-1]:
            collapsed.append(arc_id)
    return tuple(collapsed)


def reconstruct_arc_ids(
    candidates_by_observation: Sequence[Sequence[CandidateState]], decoded: DecodedPath
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map decoded state indices to per-observation and collapsed arc IDs."""
    if not decoded.state_indices:
        if decoded.log_probability == float("-inf"):
            return (), ()
        raise ValueError("decoded path must contain at least one state")
    if len(decoded.state_indices) != len(candidates_by_observation):
        raise ValueError("decoded states must align with observation candidate rows")

    matched: list[int] = []
    for row, state_index in zip(candidates_by_observation, decoded.state_indices):
        if not 0 <= state_index < len(row):
            raise ValueError("decoded state index is outside its candidate row")
        matched.append(row[state_index].arc_id)
    return tuple(matched), collapse_consecutive_arc_ids(matched)
