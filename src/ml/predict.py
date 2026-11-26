import argparse
import json

from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.ml.dataset import (
    FEATURES,
    validate_features,
)

from src.ml.models import predict_labels

from src.ml.train import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IncidentPredictor:
    """Load a trained model and predict IT incidents."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
    ):

        if not model_path.exists():

            raise FileNotFoundError(
                "Model not found. Run the training pipeline first."
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "Model metadata not found"
            )

        self.metadata = json.loads(

            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if self.metadata["features"] != FEATURES:

            raise ValueError(
                "Model feature contract does not match"
            )

        if self.metadata["sklearn_version"] != sklearn.__version__:

            raise RuntimeError(
                "Scikit-learn version mismatch. "
                "Retrain the model in this environment."
            )

        if (

            file_sha256(model_path)

            != self.metadata["model_sha256"]
        ):

            raise ValueError(
                "Model file fingerprint does not match"
            )

        self.model_name = self.metadata[
            "model_name"
        ]

        if self.model_name not in {
            "logistic_regression",
            "random_forest",
            "isolation_forest",
        }:

            raise ValueError(
                "Unsupported model type"
            )

        # Only load trusted local model files.

        self.model = joblib.load(
            model_path
        )

    def predict(
        self,
        event: dict,
    ) -> dict:
        """Predict whether a new observation is an incident."""

        df = pd.DataFrame([event])

        X = validate_features(df)

        prediction = int(

            predict_labels(

                self.model_name,

                self.model,

                X,

            )[0]
        )

        result = {

            "model": self.model_name,

            "prediction": prediction,

            "status": (

                "INCIDENT"

                if prediction == 1

                else "NORMAL"
            ),
        }

        if hasattr(
            self.model,
            "predict_proba",
        ):

            classes = list(
                self.model.classes_
            )

            incident_index = classes.index(1)

            probability = self.model.predict_proba(
                X
            )[0][incident_index]

            result["incident_probability"] = float(
                probability
            )

        elif self.model_name == "isolation_forest":

            result["anomaly_score"] = float(

                -self.model.decision_function(
                    X
                )[0]
            )

        return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--cpu",

        type=float,

        required=True,
    )

    parser.add_argument(

        "--memory",

        type=float,

        required=True,
    )

    parser.add_argument(

        "--error-rate",

        type=float,

        required=True,
    )

    parser.add_argument(

        "--latency",

        type=float,

        required=True,
    )

    args = parser.parse_args()

    predictor = IncidentPredictor(

        model_path=(

            PROJECT_ROOT

            / "models/incident_model.joblib"
        ),

        metadata_path=(

            PROJECT_ROOT

            / "models/model_metadata.json"
        ),
    )

    event = {

        "cpu_pct": args.cpu,

        "memory_pct": args.memory,

        "error_rate": args.error_rate,

        "p95_latency_ms": args.latency,
    }

    result = predictor.predict(event)

    print(
        json.dumps(result, indent=4)
    )


if __name__ == "__main__":
    main()