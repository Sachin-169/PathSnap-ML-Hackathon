# Hackathon Progress

## Member 3: HMM + Viterbi

### Completed
- Checkpoint 1: Added dependency-free HMM data structures and the initial module contract in `route_reconstruction/types.py`.
- Checkpoint 2: Added configurable Gaussian distance likelihoods and support for calibrated candidate probabilities.
- Checkpoint 3: Added directed shortest-path distance between candidate positions and an exponential route/GPS distance-consistency likelihood.
- Checkpoint 4: Added stable log emission/transition likelihoods, log-sum-exp normalization, and uniform or caller-supplied initial priors.
- Checkpoint 5: Added the variable-state Viterbi forward recurrence and predecessor table in log space.
- Checkpoint 6: Added terminal-state selection, backtracking, and per-observation/collapsed arc-sequence reconstruction.
- Checkpoint 7: Added trip-level segmentation and recovery at invalid rows, missing candidates, timestamp gaps, and infeasible road transitions.
- Checkpoint 8: Added schema/configuration validation, malformed graph checks, and explicit empty-input handling.
- Checkpoint 9: Added a standard-library automated test suite using synthetic graph and trip fixtures.
- Checkpoint 10: Documented the HMM math, data contract, configuration, Member 4 call/output, limitations, and integration tasks.
- Added ignore rules for local hackathon archives, raw trip data files, and macOS metadata.
- Post-merge integration: Added `build_map_matching_input` for current Member 1 DataFrame schemas and flat candidate rows; exported it as a public API and documented per-trip use and the optional calibrated-probability input.
- Added synthetic adapter tests for the current field mappings, missing candidate observations, probability row alignment, DataFrame node indexes, and invalid input handling.

### Pending
- Member 2: persist the chosen final model and provide an inference path that calculates the same feature columns for unlabeled candidate rows; then supply aligned calibrated probabilities if the HMM should use the model emission.
- Member 4: add the API/UI and connect it to the per-trip adapter and `match_trip` interface.
- End-to-end: validate against the final merged interfaces and an allowed held-out partition after those pieces are available.

### Interface status
- Current Member 1 preprocessing code is present and its GPS, node, road arc, and candidate columns are supported by `build_map_matching_input`. The adapter consumes one trip at a time and keeps node/arc IDs intact.
- Member 2 training code and CV results are present, but no saved final estimator or unlabeled inference feature path is present. The adapter's optional classifier probability argument is ready once an aligned probability vector is available.
- Member 4 API/UI implementation is not present yet. The stable HMM call is `match_trip(build_map_matching_input(...))`.

### Validation
- Checkpoint 1 import/structure smoke check: passed using the public data structures.
- Checkpoint 2 numerical smoke checks: passed (Gaussian reference value, distance monotonicity, classifier-probability selection, and invalid inputs).
- Checkpoint 3 synthetic graph checks: passed (directed path distance, along-arc positions, geodesic distance, and disconnected candidate scoring).
- Checkpoint 4 probability checks: passed (extreme-distance log likelihood, normalization, uniform/weighted priors, zero probability, and impossible transitions).
- Checkpoint 5 dynamic-programming checks: passed (known two-step lattice, expected scores/backpointers, and forbidden transitions).
- Checkpoint 6 backtracking checks: passed (known best path, no-path behavior, tie selection, and repeated-arc collapse).
- Checkpoint 7 synthetic trip checks: passed (connected path, long timestamp gap, missing candidates, and disconnected roads).
- Checkpoint 8 edge-case checks: passed (empty and single-point trips, missing candidates, invalid observation values, graph/candidate mismatches, and invalid configuration values).
- Checkpoints 1–10 prior validation: 17 synthetic unit tests passed, with compile checks, public-export checks, and conditional-transition normalization.
- Post-merge adapter validation: run the full suite after integration changes; results are recorded with the integration checkpoint commit.
