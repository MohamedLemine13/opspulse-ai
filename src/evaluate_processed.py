import pandas as pd

from pathlib import Path

from src.evaluate import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def adapt_events(df: pd.DataFrame) -> list[dict]:
    """
    Adapt processed monitoring records to
    the Stage 1 detector's input format.
    """

    adapted_df = df.rename(
        columns={
            "p95_latency_ms": "latency_ms",
            "cpu_pct": "cpu_usage",
        }
    )

    columns = [
        "event_id",
        "error_rate",
        "latency_ms",
        "cpu_usage",
        "incident_label",
    ]

    return adapted_df[
        columns
    ].to_dict(orient="records")


def main():

    data_path = (
        PROJECT_ROOT
        / "data/processed/events.parquet"
    )

    df = pd.read_parquet(data_path)

    events = adapt_events(df)

    results = evaluate(events)

    print("=" * 45)
    print("BASELINE ON PROCESSED DATASET")
    print("=" * 45)

    print(f"Total events:    {results['total']}")

    print(f"True positives:  {results['tp']}")

    print(f"True negatives:  {results['tn']}")

    print(f"False positives: {results['fp']}")

    print(f"False negatives: {results['fn']}")

    print("-" * 45)

    print(
        f"Precision:       "
        f"{results['precision']:.2%}"
    )

    print(
        f"Recall:          "
        f"{results['recall']:.2%}"
    )

    print(
        f"Accuracy:        "
        f"{results['accuracy']:.2%}"
    )

    print(
        f"F1-score:        "
        f"{results['f1_score']:.2%}"
    )


if __name__ == "__main__":
    main()