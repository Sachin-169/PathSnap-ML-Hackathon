# PathSnap ML Hackathon

GPS trajectory map matching with a trained candidate-road classifier, HMM emissions and transitions, Viterbi route reconstruction, a FastAPI service, and a browser evaluator.

## Install

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tune and save the classifier

Bayesian optimization uses Optuna's TPE sampler and five-fold `GroupKFold` split by trip. It tunes the HistGradientBoosting model for candidate-level macro F1, reports observation-level Top-1 as a second metric, then fits the selected configuration on all labeled training features. It writes tuning results beneath `results/` and the model plus metadata beneath `artifacts/`.

```bash
python models/bayesian_tuning.py --data-zip "$HOME/Downloads/results.zip" --trials 10
```

Alternatively, pass `--data results/candidate_features_trainable.csv` if the CSV is already extracted. Do not commit raw dataset archives or extracted trip files.

## Run the two-laptop demo

Laptop 1 (API and model):

```bash
python api/build_trip_registry.py --student-zip "hackathon_materials/Student folder.zip" --output artifacts/demo_trips.json
export PATHSNAP_TRIPS_JSON="$PWD/artifacts/demo_trips.json"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI), from a second terminal in the project root:

```bash
cd ui
python3 -m http.server 8080 --bind 0.0.0.0
```

Open `http://localhost:8080` on Laptop 2. Enter Laptop 1's Wi-Fi IP with port 8000 in the API address field, load trips, select one, then show its route. The UI shows the road network, GPS points, and HMM/Viterbi predicted route.

## Make a standalone Folium map

After the registry and trained classifier exist:

```bash
python visualizations/generate_folium_map.py 00000009
```

This writes `artifacts/maps/trip_00000009.html`. To add the held-out reference route as an optional, initially hidden comparison layer, pass `--reference-zip "hackathon_materials/Student folder.zip"`. Reference labels are read only for this display layer, after inference, and are not used by the model or decoder.

## Validate

```bash
python -m unittest discover -s tests -v
```

See [`route_reconstruction/README.md`](route_reconstruction/README.md), [`api/README.md`](api/README.md), [`models/README.md`](models/README.md), and [`visualizations/README.md`](visualizations/README.md) for the module details and exact inputs/outputs.
