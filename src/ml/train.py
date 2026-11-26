import argparse
import hashlib
import json
import platform

from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.ml.dataset import (
    FEATURES,
    TARGET,
    load_dataset,
    chronological_split,
    get_X_y,
    summarize_dataset,
)

from src.ml.models import (
    RANDOM_STATE,
    create_models,
    fit_model,
    predict_labels,
    baseline_predictions,
)

from src.ml.metrics import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_json(data: dict, path: Path):
    """Save a dictionary as a JSON document."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(data, indent=4),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    """Calculate a reproducible fingerprint of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(65536),
            b"",
        ):

            digest.update(block)

    return digest.hexdigest()


def select_best_model(
    validation_results: pd.DataFrame,
) -> str:
    """Select the best ML model using validation metrics."""

    candidates = validation_results[
        validation_results["model"] != "rule_based"
    ]

    if candidates.empty:
        raise ValueError("No ML candidates available")

    ranked = candidates.sort_values(
        by=[
            "f1_score",
            "recall",
            "precision",
            "model",
        ],

        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    return str(
        ranked.iloc[0]["model"]
    )


def run_training(
    input_path: Path,
    models_dir: Path,
    reports_dir: Path,
) -> dict:
    """Execute the complete ML training workflow."""

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # 1. Load and split the dataset
    # -----------------------------------------

    df = load_dataset(input_path)

    train, validation, test = chronological_split(df)

    X_train, y_train = get_X_y(train)

    X_validation, y_validation = get_X_y(validation)

    X_test, y_test = get_X_y(test)

    dataset_summary = {

        "full_dataset": summarize_dataset(df),

        "train": summarize_dataset(train),

        "validation": summarize_dataset(validation),

        "test": summarize_dataset(test),
    }

    save_json(
        dataset_summary,

        reports_dir / "dataset_summary.json",
    )

    print("=" * 50)
    print("OPSPULSE AI - MACHINE LEARNING TRAINING")
    print("=" * 50)

    print(f"Total observations: {len(df)}")

    print(f"Training:           {len(train)}")

    print(f"Validation:         {len(validation)}")

    print(f"Testing:            {len(test)}")

    print(f"Features:           {FEATURES}")

    # -----------------------------------------
    # 2. Evaluate the rule-based baseline
    # -----------------------------------------

    baseline_val = baseline_predictions(
        X_validation
    )

    validation_results = [
        {
            "model": "rule_based",

            **calculate_metrics(
                y_validation,
                baseline_val,
            ),
        }
    ]

    # -----------------------------------------
    # 3. Train and validate ML candidates
    # -----------------------------------------

    models = create_models()

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model = fit_model(
            name,

            model,

            X_train,

            y_train,
        )

        trained_models[name] = model

        predictions = predict_labels(
            name,

            model,

            X_validation,
        )

        metrics = calculate_metrics(
            y_validation,

            predictions,
        )

        validation_results.append({

            "model": name,

            **metrics,
        })

        print(
            f"Validation F1: "
            f"{metrics['f1_score']:.4f}"
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df = validation_df.sort_values(
        "f1_score",
        ascending=False,
    )

    validation_df.to_csv(

        reports_dir / "validation_metrics.csv",

        index=False,
    )

    print("\nVALIDATION RESULTS")

    print(
        validation_df.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------
    # 4. Select the best ML model
    # -----------------------------------------

    best_name = select_best_model(
        validation_df
    )

    best_model = trained_models[best_name]

    print(
        f"\nSelected ML model: {best_name}"
    )

    baseline_validation_f1 = float(

        validation_df.loc[
            validation_df["model"] == "rule_based",

            "f1_score",
        ].iloc[0]
    )

    selected_validation_f1 = float(

        validation_df.loc[
            validation_df["model"] == best_name,

            "f1_score",
        ].iloc[0]
    )

    # -----------------------------------------
    # 5. Final held-out test evaluation
    # -----------------------------------------

    baseline_test = baseline_predictions(
        X_test
    )

    model_test = predict_labels(
        best_name,

        best_model,

        X_test,
    )

    baseline_metrics = calculate_metrics(
        y_test,

        baseline_test,
    )

    selected_metrics = calculate_metrics(
        y_test,

        model_test,
    )

    test_results = pd.DataFrame([

        {
            "model": "rule_based",

            **baseline_metrics,
        },

        {
            "model": best_name,

            **selected_metrics,
        },
    ])

    test_results.to_csv(

        reports_dir / "test_metrics.csv",

        index=False,
    )

    print("\nHELD-OUT TEST RESULTS")

    print(

        test_results.round(4).to_string(
            index=False
        )
    )

    # -----------------------------------------
    # 6. Save individual predictions
    # -----------------------------------------

    predictions_df = test[

        [
            "event_id",
            "timestamp",
            *FEATURES,
            TARGET,
        ]

    ].copy()

    predictions_df["baseline_prediction"] = (
        baseline_test
    )

    predictions_df["model_prediction"] = (
        model_test
    )

    predictions_df["model_correct"] = (

        predictions_df[TARGET]

        == predictions_df["model_prediction"]
    )

    predictions_df.to_csv(

        reports_dir / "test_predictions.csv",

        index=False,
    )

    # -----------------------------------------
    # 7. Save the selected model
    # -----------------------------------------

    model_path = (
        models_dir / "incident_model.joblib"
    )

    joblib.dump(
        best_model,

        model_path,

        compress=3,
    )

    # -----------------------------------------
    # 8. Save model metadata
    # -----------------------------------------

    metadata = {

        "model_name": best_name,

        "model_file": model_path.name,

        "features": FEATURES,

        "target": TARGET,

        "class_mapping": {

            "0": "NORMAL",

            "1": "INCIDENT",
        },

        "training_strategy": (
            "unsupervised"
            if best_name == "isolation_forest"
            else "supervised"
        ),

        "random_state": RANDOM_STATE,

        "dataset_file": input_path.name,

        "dataset_sha256": file_sha256(
            input_path
        ),

        "model_sha256": file_sha256(
            model_path
        ),

        "training_rows": len(train),

        "validation_rows": len(validation),

        "test_rows": len(test),

        "split_strategy": "chronological_70_15_15",

        "dataset_summary": dataset_summary,

        "python_version": platform.python_version(),

        "sklearn_version": sklearn.__version__,

        "pandas_version": pd.__version__,

        "joblib_version": joblib.__version__,

        "selection_metric": "f1_score",

        "validation_f1_score": (
            selected_validation_f1
        ),

        "baseline_validation_f1_score": (
            baseline_validation_f1
        ),

        "beats_baseline_on_validation": (

            selected_validation_f1

            > baseline_validation_f1
        ),

        "test_metrics": selected_metrics,

        "baseline_test_metrics": baseline_metrics,
    }

    save_json(

        metadata,

        models_dir / "model_metadata.json",
    )

    print("\nMODEL SAVED")

    print(model_path)

    print("\nTRAINING COMPLETED SUCCESSFULLY")

    return metadata


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--input",

        type=Path,

        default=(

            PROJECT_ROOT

            / "data/processed/events.parquet"
        ),
    )

    args = parser.parse_args()

    run_training(

        input_path=args.input,

        models_dir=PROJECT_ROOT / "models",

        reports_dir=PROJECT_ROOT / "reports/ml",
    )


if __name__ == "__main__":
    main()