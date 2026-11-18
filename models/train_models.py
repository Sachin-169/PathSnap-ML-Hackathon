
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "results" / "candidate_features_trainable.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
KNN_TRAIN_SAMPLE_SIZE = 10000
KNN_VAL_SAMPLE_SIZE = 2000

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

TARGET_COLUMN = "label"
GROUP_COLUMN = "trip_id"


# --------------------------------------------------
# LOAD ONLY REQUIRED COLUMNS
# --------------------------------------------------

def load_dataset():

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, GROUP_COLUMN]
    )

    print("Loading training dataset...")

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        dtype={
            "trip_id": "int32",
            "label": "int8",
        },
    )

    X = df[FEATURE_COLUMNS].astype("float32")
    y = df[TARGET_COLUMN].astype("int8")
    groups = df[GROUP_COLUMN].astype("int32")

    print("Dataset shape:", X.shape)
    print("Number of trips:", groups.nunique())
    print("\nClass distribution:")
    print(y.value_counts())

    return X, y, groups


# --------------------------------------------------
# DEFINE FIVE MODELS
# --------------------------------------------------

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=4,
            random_state=RANDOM_STATE,
        ),

        "K-Nearest Neighbors": make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        algorithm="brute",
        n_jobs=4,
    ),
),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.1,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------
# GROUP-AWARE CROSS VALIDATION
# --------------------------------------------------


def evaluate_models(X, y, groups):

    cv = GroupKFold(n_splits=N_SPLITS)
    models = get_models()

    results = []

    # Save each completed fold immediately.
    fold_path = RESULTS_DIR / "model_cv_fold_results.csv"

    # Remove results from an older experiment.
    if fold_path.exists():
        fold_path.unlink()

    for model_name, model in models.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("=" * 65)

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            assert train_groups.isdisjoint(val_groups)

            # Use smaller samples for KNN.
            if model_name == "K-Nearest Neighbors":

                train_sample_indices = X_train.sample(
                    n=min(KNN_TRAIN_SAMPLE_SIZE, len(X_train)),
                    random_state=RANDOM_STATE + fold,
                ).index

                val_sample_indices = X_val.sample(
                    n=min(KNN_VAL_SAMPLE_SIZE, len(X_val)),
                    random_state=RANDOM_STATE + fold,
                ).index

                X_train = X_train.loc[train_sample_indices]
                y_train = y_train.loc[train_sample_indices]

                X_val = X_val.loc[val_sample_indices]
                y_val = y_val.loc[val_sample_indices]

                print(
                    f"KNN sample sizes: "
                    f"Training={len(X_train)}, "
                    f"Validation={len(X_val)}"
                )

            current_model = clone(model)

            start_time = time.perf_counter()

            current_model.fit(X_train, y_train)

            training_time = time.perf_counter() - start_time

            predictions = current_model.predict(X_val)
            probabilities = current_model.predict_proba(X_val)

            fold_result = {
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y_val, predictions),
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "log_loss": log_loss(
                    y_val,
                    probabilities,
                    labels=[0, 1],
                ),
                "training_time_seconds": training_time,
                "train_trips": len(train_groups),
                "validation_trips": len(val_groups),
            }

            results.append(fold_result)

            # Save immediately after this fold.
            pd.DataFrame([fold_result]).to_csv(
                fold_path,
                mode="a",
                header=not fold_path.exists(),
                index=False,
            )

            print(
                f"Fold {fold}: "
                f"F1={fold_result['f1']:.4f}, "
                f"Precision={fold_result['precision']:.4f}, "
                f"Recall={fold_result['recall']:.4f}, "
                f"Time={training_time:.2f}s"
            )

            print("Fold result saved.")

    return pd.DataFrame(results)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    fold_path = RESULTS_DIR / "model_cv_fold_results.csv"

    results.to_csv(fold_path, index=False)

    summary = (
        results.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            log_loss_mean=("log_loss", "mean"),
            training_time_mean=("training_time_seconds", "mean"),
            total_training_time=("training_time_seconds", "sum"),
        )
        .reset_index()
        .sort_values("f1_mean", ascending=False)
    )

    summary_path = RESULTS_DIR / "model_cv_summary.csv"

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)

    print(summary.to_string(index=False))

    print("\nSaved fold results:", fold_path)
    print("Saved summary:", summary_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    X, y, groups = load_dataset()

    results = evaluate_models(X, y, groups)

    save_results(results)

    print("\nInitial model comparison completed.")


if __name__ == "__main__":
    main()
