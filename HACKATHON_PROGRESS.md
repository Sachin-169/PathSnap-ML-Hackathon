# PathSnap integration progress

## Member 3: HMM + Viterbi

The HMM/Viterbi implementation is in `route_reconstruction/`: typed trip and road-network structures, Gaussian or classifier-scored emissions, directed graph transition probabilities, log-space Viterbi decoding, backtracking, and recovery at GPS gaps or infeasible transitions. `build_map_matching_input` adapts the current preprocessing schemas.

## Completed integration

- Built label-free inference features matching the eight model columns from the training feature pipeline.
- Added Optuna TPE Bayesian hyperparameter search for HistGradientBoosting, using five-fold trip-grouped CV and candidate-level macro F1 as the objective; the run also reports observation-level Top-1.
- Added final model export plus tuning trial and summary files.
- Added a FastAPI service that scores aligned candidate features and sends the probabilities to HMM/Viterbi.
- Added a student-test registry builder that reads track, node, and arc files without loading reference route labels.
- Added the evaluator UI integration and a separate Folium HTML map generator with network, GPS, predicted route, endpoints, and optional reference comparison layer.
- Added run instructions in the root and per-module READMEs.

## Validation status

- Bayesian search: 10 trials requested; 6 completed fully and 4 were pruned. Best grouped macro F1 was 0.8715 vs. 0.8691 for the baseline. Observation Top-1 was 0.7880 vs. 0.7868. These metrics are recorded in `results/bayesian_optimization_summary.json`; neither should be described as route accuracy.
- Full synthetic suite: 25 tests passed.
- End-to-end API smoke check: health and trip-list endpoints returned HTTP 200; trip `00000009` returned a predicted route for 1,049 observations and 13,104 road arcs (432 predicted arcs across 2 decoded segments).
- Folium smoke check: generated `artifacts/maps/trip_00000009.html` with road-network, GPS, predicted-route, start/end, and optional hidden reference-route layers.
- Python compile check: passed for API, preprocessing, models, route reconstruction, visualizations, and tests.

## Local-only inputs and generated files

Dataset archives, raw trajectory files, the student-trip registry, HTML maps, and training CSVs remain local and are ignored by Git. The final classifier artifact and tuning summary are produced by the documented tuning run and can be used by the API on Laptop 1.
