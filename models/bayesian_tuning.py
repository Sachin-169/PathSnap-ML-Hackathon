"""Group-aware Bayesian tuning and final estimator export for HistGradientBoosting.

The objective is five-fold GroupKFold macro F1. Trip IDs are kept intact across
folds. The script also records observation-level Top-1 accuracy, fits the best
configuration on all training rows, and saves a joblib artifact for the API.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
    "distance_m",
    "projection_fraction",
    "road_length_m",
    "speed_mps",
    "time_difference_s",
    "heading_difference",
    "distance_to_start_m",
    "distance_to_end_m",
]
METADATA_COLUMNS = ["trip_id", "observation_id", "arc_id", "reference_arc_id"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["label"] + METADATA_COLUMNS


def load_data(data_path: Path, archive_path: Path | None) -> pd.DataFrame:
    dtypes: dict[str, str] = {column: "float32" for column in FEATURE_COLUMNS}
    dtypes.update(
        {
            "trip_id": "int32",
            "observation_id": "int32",
            "arc_id": "int32",
            "reference_arc_id": "float32",
            "label": "int8",
        }
    )
    if data_path.is_file():
        frame = pd.read_csv(data_path, usecols=REQUIRED_COLUMNS, dtype=dtypes)
    elif archive_path is not None and archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            member = "results/candidate_features_trainable.csv"
            if member not in archive.namelist():
                raise FileNotFoundError(f"{member} was not found in {archive_path}")
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, usecols=REQUIRED_COLUMNS, dtype=dtypes)
    else:
        raise FileNotFoundError(
            f"Training features not found at {data_path}. Supply --data-zip "
            "pointing to the results.zip containing candidate_features_trainable.csv."
        )
    if frame.empty or frame["trip_id"].nunique() < 5:
        raise ValueError("Need non-empty training data from at least five trips")
    return frame


def _new_model(params: dict[str, Any], seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **params,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )


def evaluate_cv(
    params: dict[str, Any], frame: pd.DataFrame, *, report_trial=None
) -> tuple[float, float]:
    X = frame[FEATURE_COLUMNS]
    y = frame["label"]
    groups = frame["trip_id"]
    metadata = frame[METADATA_COLUMNS]
    cv = GroupKFold(n_splits=5)
    macro_scores: list[float] = []
    top1_scores: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = _new_model(params, seed=42 + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[valid_idx])
        macro_scores.append(
            f1_score(y.iloc[valid_idx], predictions, average="macro", zero_division=0)
        )

        probability_column = list(model.classes_).index(1)
        positive_probability = model.predict_proba(X.iloc[valid_idx])[:, probability_column]
        valid_metadata = metadata.iloc[valid_idx].reset_index(drop=True).copy()
        valid_metadata["positive_probability"] = positive_probability
        best_rows = valid_metadata.loc[
            valid_metadata.groupby(
                ["trip_id", "observation_id"], sort=False
            )["positive_probability"].idxmax()
        ]
        known_reference = best_rows["reference_arc_id"].notna()
        top1 = (
            best_rows.loc[known_reference, "arc_id"].to_numpy()
            == best_rows.loc[known_reference, "reference_arc_id"].to_numpy()
        )
        top1_scores.append(float(np.mean(top1)) if len(top1) else float("nan"))

        if report_trial is not None:
            report_trial.report(float(np.mean(macro_scores)), step=fold)
            if report_trial.should_prune():
                raise optuna.TrialPruned()
        print(
            f"    fold {fold}/5: macro_f1={macro_scores[-1]:.4f}, "
            f"top1={top1_scores[-1]:.4f}",
            flush=True,
        )
        del model, predictions, positive_probability, valid_metadata, best_rows
        gc.collect()

    return float(np.mean(macro_scores)), float(np.mean(top1_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "results" / "candidate_features_trainable.csv",
    )
    parser.add_argument(
        "--data-zip",
        type=Path,
        default=None,
        help="Optional results.zip containing results/candidate_features_trainable.csv",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame = load_data(args.data.expanduser(), args.data_zip.expanduser() if args.data_zip else None)
    print(
        f"Loaded {len(frame):,} candidate rows across "
        f"{frame['trip_id'].nunique()} trips. Optimizing macro F1.",
        flush=True,
    )

    baseline_params = {
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
        "max_bins": 255,
    }
    print("Evaluating the untuned reference configuration...", flush=True)
    baseline_macro, baseline_top1 = evaluate_cv(baseline_params, frame)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_iter": trial.suggest_int("max_iter", 80, 240, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.025, 0.2, log=True
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-5, 20.0, log=True
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 100, log=True
            ),
            "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
        }
        print(f"Trial {trial.number + 1}/{args.trials}: {params}", flush=True)
        macro_f1, top1 = evaluate_cv(params, frame, report_trial=trial)
        trial.set_user_attr("cv_top1", top1)
        print(
            f"  trial complete: macro_f1={macro_f1:.4f}, top1={top1:.4f}",
            flush=True,
        )
        return macro_f1

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best_params = dict(study.best_params)
    tuned_top1 = float(study.best_trial.user_attrs.get("cv_top1", float("nan")))
    print("Fitting selected model on all available training rows...", flush=True)
    final_model = _new_model(best_params, seed=args.seed)
    final_model.fit(frame[FEATURE_COLUMNS], frame["label"])

    results_dir = ROOT / "results"
    artifacts_dir = ROOT / "artifacts"
    results_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    trials_path = results_dir / "bayesian_optimization_trials.csv"
    trial_rows = []
    for trial in study.trials:
        trial_rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "cv_macro_f1": trial.value,
                "cv_top1": trial.user_attrs.get("cv_top1"),
                **trial.params,
            }
        )
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)

    summary = {
        "model": "HistGradientBoostingClassifier",
        "objective": "5-fold GroupKFold candidate-level macro F1",
        "trials_requested": args.trials,
        "trials_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "baseline_cv_macro_f1": baseline_macro,
        "baseline_cv_top1": baseline_top1,
        "best_cv_macro_f1": float(study.best_value),
        "best_cv_top1": tuned_top1,
        "best_params": best_params,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(frame)),
        "training_trips": int(frame["trip_id"].nunique()),
    }
    model_path = artifacts_dir / "final_classifier.joblib"
    metadata_path = artifacts_dir / "final_classifier_metadata.json"
    joblib.dump(final_model, model_path, compress=3)
    metadata_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([summary]).to_json(
        results_dir / "bayesian_optimization_summary.json", orient="records", indent=2
    )
    print(f"Saved trial results: {trials_path}")
    print(f"Saved final estimator: {model_path}")
    print(f"Saved tuning metadata: {metadata_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
