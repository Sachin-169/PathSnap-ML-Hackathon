"""Viterbi forward dynamic programming over log probabilities."""

from __future__ import annotations

import math
from numbers import Real
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ViterbiLattice:
    """Best log score and predecessor index for every state at every time."""

    scores: tuple[tuple[float, ...], ...]
    backpointers: tuple[tuple[int | None, ...], ...]


def _validate_log_value(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value) or value == math.inf:
        raise ValueError(f"{name} must be finite or negative infinity")


def viterbi_forward(
    initial_log_probabilities: Sequence[float],
    emission_log_probabilities: Sequence[Sequence[float]],
    transition_log_probabilities: Sequence[Sequence[Sequence[float]]],
) -> ViterbiLattice:
    """Build the Viterbi score and predecessor tables.

    ``transition_log_probabilities[t - 1][i][j]`` is the log probability of
    moving from state ``i`` at time ``t - 1`` to state ``j`` at time ``t``.
    Stable iteration order resolves exact ties in favor of the lowest prior
    state index.
    """
    if not emission_log_probabilities:
        raise ValueError("at least one observation is required")
    if len(initial_log_probabilities) != len(emission_log_probabilities[0]):
        raise ValueError("initial probabilities must align with first emissions")
    if len(transition_log_probabilities) != len(emission_log_probabilities) - 1:
        raise ValueError("one transition matrix is required between each observation")

    for value in initial_log_probabilities:
        _validate_log_value(value, "initial log probability")
    for row in emission_log_probabilities:
        if not row:
            raise ValueError("each observation must have at least one candidate state")
        for value in row:
            _validate_log_value(value, "emission log probability")

    first_scores = tuple(
        initial + emission
        for initial, emission in zip(initial_log_probabilities, emission_log_probabilities[0])
    )
    score_rows: list[tuple[float, ...]] = [first_scores]
    pointer_rows: list[tuple[int | None, ...]] = [tuple(None for _ in first_scores)]

    for time_index, transition_matrix in enumerate(transition_log_probabilities, start=1):
        previous_scores = score_rows[-1]
        current_emissions = emission_log_probabilities[time_index]
        if len(transition_matrix) != len(previous_scores):
            raise ValueError("transition matrix row count must match previous states")
        if any(len(row) != len(current_emissions) for row in transition_matrix):
            raise ValueError("transition matrix column count must match current states")
        for row in transition_matrix:
            for value in row:
                _validate_log_value(value, "transition log probability")

        current_scores: list[float] = []
        current_pointers: list[int | None] = []
        for current_index, emission_score in enumerate(current_emissions):
            best_score = -math.inf
            best_previous: int | None = None
            for previous_index, previous_score in enumerate(previous_scores):
                candidate_score = previous_score + transition_matrix[previous_index][current_index]
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous_index
            current_scores.append(best_score + emission_score)
            current_pointers.append(best_previous)
        score_rows.append(tuple(current_scores))
        pointer_rows.append(tuple(current_pointers))

    return ViterbiLattice(tuple(score_rows), tuple(pointer_rows))


@dataclass(frozen=True)
class DecodedPath:
    """Recovered state-index sequence and its best log probability."""

    state_indices: tuple[int, ...]
    log_probability: float


def viterbi_backtrack(lattice: ViterbiLattice) -> DecodedPath:
    """Select the best terminal state and follow predecessor pointers backward.

    If every terminal state is impossible, return an empty path with a
    negative-infinity score; callers can then recover or report no match.
    """
    if not lattice.scores or len(lattice.scores) != len(lattice.backpointers):
        raise ValueError("lattice must contain matching score and pointer rows")
    for time_index, (scores, pointers) in enumerate(zip(lattice.scores, lattice.backpointers)):
        if not scores or len(scores) != len(pointers):
            raise ValueError(f"lattice row {time_index} has inconsistent dimensions")
        for score in scores:
            _validate_log_value(score, "Viterbi score")
    final_scores = lattice.scores[-1]
    if not final_scores:
        raise ValueError("lattice must contain at least one terminal state")
    best_state = max(range(len(final_scores)), key=final_scores.__getitem__)
    best_score = final_scores[best_state]
    if best_score == -math.inf:
        return DecodedPath((), -math.inf)

    state_indices = [best_state]
    for time_index in range(len(lattice.scores) - 1, 0, -1):
        pointers = lattice.backpointers[time_index]
        if len(pointers) != len(lattice.scores[time_index]):
            raise ValueError("backpointer row does not align with score row")
        previous_state = pointers[state_indices[-1]]
        if previous_state is None:
            raise ValueError("reachable state is missing its predecessor")
        if not 0 <= previous_state < len(lattice.scores[time_index - 1]):
            raise ValueError("backpointer refers to an invalid previous state")
        state_indices.append(previous_state)
    state_indices.reverse()
    return DecodedPath(tuple(state_indices), best_score)
