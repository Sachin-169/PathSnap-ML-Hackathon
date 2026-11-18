# PathSnap HMM + Viterbi module

This module matches each GPS observation to a sequence of candidate road arcs. It exposes the trip-level function `match_trip` and uses only the Python standard library.

## Data contract

The supplied dataset documentation describes these headerless, whitespace-separated files, grouped by trip:

- `.track`: longitude, latitude, timestamp in seconds.
- `.nodes`: longitude, latitude; the row index is the node ID.
- `.arcs`: from-node ID, to-node ID; the row index is the arc ID.
- `.route`: chronological reference arc IDs. Route rows are not assumed to align one-to-one with GPS observations.

The typed contract below remains the HMM's internal API. The current preprocessing schema has since been added in `data_processing/`; `build_map_matching_input` adapts its GPS, projected nodes, road arcs, and flat candidate records into this contract. The adapter is deliberately confined to this package and does not change upstream team modules.

| Type | Required data |
|---|---|
| `GPSObservation` | `timestamp_s`, `longitude`, `latitude` for one timestamp-ordered sample. |
| `RoadNode` | Non-negative `node_id`, longitude, latitude. |
| `RoadArc` | Non-negative `arc_id`, directed `from_node_id`, `to_node_id`, and optional `length_m`. |
| `CandidateState` | `arc_id`, observation-to-road `distance_m`, optional `fraction_along` in `[0, 1]`, optional calibrated `classifier_probability` in `[0, 1]`. |
| `RoadNetwork` | `nodes` and `arcs` mappings keyed by their IDs. |
| `MapMatchingInput` | One observation row and exactly one candidate row per observation, plus the trip road network. |

Member 1 currently provides flat candidate rows keyed by `observation_id`, with `arc_id`, `distance_m`, and `projection_fraction`. `build_map_matching_input` groups them into one candidate list per cleaned GPS row, maps `timestamp` to `timestamp_s`, maps `start_node`/`end_node` to the internal arc fields, and preserves `length_m`. It accepts pandas DataFrames or sequences of row mappings. It uses node DataFrame indexes as node IDs when no `node_id` column exists. Pass data for one trip at a time; candidate observation IDs must refer to that trip's cleaned GPS row order. Missing candidate observations are represented as empty candidate lists and decoded as unmatched. The current candidate generator raises when the entire candidate table is empty, so that upstream condition must be handled before calling the adapter.

Example:

```python
from route_reconstruction import build_map_matching_input, match_trip

trip_input = build_map_matching_input(
    cleaned_gps, projected_nodes, road_network, candidates,
    classifier_probabilities=candidate_positive_probabilities,  # optional
)
result = match_trip(trip_input)
```

If supplied, `candidate_positive_probabilities` must contain one calibrated positive-class probability per candidate row, in exactly the same row order as the candidate table. The adapter attaches probabilities before grouping by observation. Do not pass feature rows after filtering, shuffling, or a train/test transformation unless the probabilities are aligned to that exact table.

The current preprocessing candidate function can produce an empty candidate table only by raising an exception; the adapter itself supports an empty table and creates empty candidate lists. If the road network has arcs removed during validity filtering, retain its `arc_id` values as generated; do not renumber arcs independently of candidates.

Member 2 may populate `classifier_probability` with a calibrated probability for each candidate. Do not pass logits, margins, or uncalibrated scores in this field. When it is absent, the module uses the geometric Gaussian emission model. The module does not consume `.route` labels; use permitted reference routes outside the prediction path for development evaluation.

## HMM formulation

At observation `t`, each candidate road arc is a hidden state `s_t`. The GPS coordinate and timestamp form the observation. The initial state distribution is uniform over the first observation's candidates unless caller-supplied weights are used through `initial_log_probabilities`.

For perpendicular GPS-to-road distance `d` and configurable scale `sigma`, the geometric emission density is:

```text
E_t(s) = exp(-d² / (2 sigma²)) / (sqrt(2 pi) sigma)
log E_t(s) = -log(sqrt(2 pi) sigma) - d² / (2 sigma²)
```

If `classifier_probability` is supplied, that candidate score is used as the emission likelihood instead. The score is not multiplied by the geometric density, avoiding accidental double counting of a distance feature.

For a candidate pair, let `d_net` be shortest directed road-network travel distance between the candidate positions, and `d_gps` be geodesic displacement between the observations. The unnormalized transition weight is:

```text
W(i, j) = exp(-abs(d_net - d_gps) / beta)
```

For each previous candidate `i`, weights are normalized over feasible next candidates `j` to form conditional transition probabilities. A transition is infeasible if there is no directed path or if `d_net / elapsed_seconds` exceeds the configurable maximum speed. Candidate-position distance includes the remainder of the previous arc, a shortest path between endpoint nodes, and the traversed portion of the next arc.

All calculations used by Viterbi are in log space. For state `j` at time `t`:

```text
Delta_0(j) = log(initial(j)) + log(E_0(j))
Delta_t(j) = log(E_t(j)) + max_i(Delta_(t-1)(i) + log(T_t(i, j)))
```

The decoder stores the maximizing predecessor for each state and follows those pointers backward from the best final state. Impossible probabilities are represented as negative infinity. If no complete path exists, the output remains unmatched; the module does not fabricate a route.

## Gaps and failure handling

- Invalid or missing coordinates/timestamps produce a `None` match for that observation and split the sequence.
- An empty candidate row produces a `None` match and splits the sequence.
- A non-increasing timestamp, a gap larger than `max_gap_seconds`, or a transition matrix with no feasible path starts a new independent segment.
- A road-network break therefore appears as separate `segment_ids`; it is not bridged with an invented route.
- An empty trip returns empty output sequences. A single valid observation is decoded normally.
- Candidate IDs not present in the graph, malformed graph references, invalid coordinates, and invalid configuration values raise clear `TypeError` or `ValueError` exceptions.

## Configuration

`MapMatchingConfig` defaults are starting values, not parameters fitted on this dataset:

- `emission_sigma_m=20.0`
- `transition_beta_m=50.0`
- `max_speed_mps=55.6`
- `max_gap_seconds=10.0`

Tune and compare these values using training and student-testing data only. Never tune against evaluator-only trips or labels. Record the actual values and validation results in the team report.

## Member 4 call and output

```python
from route_reconstruction import (
    CandidateState,
    GPSObservation,
    MapMatchingConfig,
    MapMatchingInput,
    RoadArc,
    RoadNetwork,
    RoadNode,
    match_trip,
)

network = RoadNetwork(
    nodes={
        0: RoadNode(0, -93.35020, 44.86495),
        1: RoadNode(1, -93.35010, 44.86495),
    },
    arcs={0: RoadArc(0, 0, 1, length_m=8.0)},
)
trip = MapMatchingInput(
    observations=[GPSObservation(1.0, -93.35019, 44.86495)],
    candidates_by_observation=[
        [CandidateState(arc_id=0, distance_m=1.2, fraction_along=0.1)]
    ],
    network=network,
)
result = match_trip(trip, MapMatchingConfig())
```

`MapMatchingResult` fields:

- `matched_arc_ids`: one arc ID or `None` per input observation.
- `route_arc_ids`: chronological arc sequence with consecutive duplicates collapsed within each decoded segment.
- `segment_ids`: one segment number or `None` per observation. A repeated arc at a segment boundary is preserved in `route_arc_ids`.
- `log_score`: summed best log score across decoded segments. It is a ranking/likelihood score, **not a calibrated confidence value**.
- `diagnostics`: human-readable notes for invalid observations, missing candidates, gaps, and route breaks.

For JSON responses, convert tuples to lists. Member 4 should keep evaluator reference labels server-side and must not send them into `MapMatchingInput` or expose them in the UI.

## Validation

Run the synthetic automated suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The suite covers emissions, initial priors, Viterbi scores/backtracking, route reconstruction, connected and disconnected roads, GPS gaps, missing candidates, over-speed transitions, and empty trips. No full-dataset performance or accuracy claim has been made; the current tests use synthetic fixtures only.

## Limitations and integration tasks

1. The Member 1 schema adapter is implemented. Confirm once more when interfaces change; the Member 2 inference artifact/probability interface and Member 4 API/UI integration are not present yet.
2. Check whether each `.arcs` row is truly directed and whether reverse directions are represented as separate rows. This implementation respects the provided direction.
3. Supply candidate along-arc fractions and realistic arc lengths where possible; midpoint and endpoint-chord fallbacks are approximations.
4. Profile shortest-path search on permitted trip graphs and tune the bounded pair cache if needed.
5. Validate probability calibration, parameters, route continuity, and inference time on training and student-testing partitions before reporting results. The current model training script evaluates cross-validation models but does not persist a final estimator, so classifier probabilities cannot yet be wired into this adapter from a saved model.
