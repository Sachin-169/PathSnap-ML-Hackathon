# FastAPI demo service

`main.py` exposes the evaluator endpoints. It loads the local, label-free trip registry from `PATHSNAP_TRIPS_JSON` and the tuned classifier from `PATHSNAP_MODEL_PATH` (defaults to `artifacts/final_classifier.joblib`). Each request scores candidate rows in their stored order, passes positive-class probabilities into the HMM adapter, and returns the reconstructed route with GPS and road geometry. Reference route labels are not served.

## Prepare Laptop 1

Install project dependencies and tune/export the model first; see the root README. Keep the supplied student archive local. Then build the API's registry:

```bash
python api/build_trip_registry.py \
  --student-zip "hackathon_materials/Student folder.zip" \
  --output artifacts/demo_trips.json
```

The builder reads `.track`, `.nodes`, and `.arcs`, then applies the current GPS cleaning, road-network construction, candidate generation, and label-free inference feature generation. It never opens `.route`. The generated registry and dataset-derived files are ignored by Git.

## Start the API

```bash
export PATHSNAP_TRIPS_JSON="$PWD/artifacts/demo_trips.json"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 1's terminal shows Uvicorn access logs for UI requests. In another terminal, `curl http://localhost:8000/api/health` checks service health. `GET /api/trips` lists trip IDs and `POST /api/trips/{trip_id}/match` runs the model and HMM/Viterbi route reconstruction.

## Laptop 2

Serve the static UI with `cd ui && python3 -m http.server 8080 --bind 0.0.0.0`, open `http://localhost:8080`, and set the API address to `http://<Laptop-1-Wi-Fi-IP>:8000`. On the same machine use `http://localhost:8000`. The API must remain running while the UI is used.

The registry schema and adapter inputs are documented in `route_reconstruction/README.md`. Candidate feature row order must remain identical to candidate row order; the builder and API validate this alignment.
