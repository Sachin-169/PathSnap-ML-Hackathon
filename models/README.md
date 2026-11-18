# Bayesian model tuning

`bayesian_tuning.py` runs Optuna's TPE Bayesian search for a `HistGradientBoostingClassifier`. Its primary objective is mean five-fold GroupKFold candidate-level macro F1, with `trip_id` as the group so rows from the same trip cannot leak between training and validation folds. The run records observation-level Top-1 as a diagnostic and compares the selected score with the reference HGB configuration.

The final selected configuration is fit on all labeled rows. The API consumes the resulting `artifacts/final_classifier.joblib` and its metadata file. The model inputs are the eight columns in `data_processing/inference_features.py`; inference feature construction deliberately excludes labels and reference route IDs.

## Run

```bash
python models/bayesian_tuning.py --data-zip "$HOME/Downloads/results.zip" --trials 10
```

The archive must contain `results/candidate_features_trainable.csv`. Alternatively, use `--data PATH_TO_CSV`. Outputs are `results/bayesian_optimization_trials.csv`, `results/bayesian_optimization_summary.json`, `artifacts/final_classifier.joblib`, and `artifacts/final_classifier_metadata.json`.

The objective is candidate classification macro F1. It is not route accuracy, and it should not be reported as “95% accuracy” unless the measured metric and held-out evaluation support that claim. The test trips are not used during tuning.
