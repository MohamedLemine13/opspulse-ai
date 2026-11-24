from pathlib import Path

from src.detector import (
    load_events,
    predict_incident,
)


def evaluate(events: list[dict]) -> dict:
    """Evaluate the rule-based incident detector."""

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for event in events:

        actual = event["incident_label"]
        predicted = predict_incident(event)

        if actual == 1 and predicted == 1:
            tp += 1

        elif actual == 0 and predicted == 0:
            tn += 1

        elif actual == 0 and predicted == 1:
            fp += 1

        elif actual == 1 and predicted == 0:
            fn += 1

    total = tp + tn + fp + fn

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
    }


def analyze_errors(events: list[dict]):
    """Display incorrectly classified events."""

    print("\nMISCLASSIFIED EVENTS")
    print("-" * 65)

    for event in events:

        actual = event["incident_label"]
        predicted = predict_incident(event)

        if actual != predicted:

            error_type = (
                "FALSE POSITIVE"
                if predicted == 1
                else "FALSE NEGATIVE"
            )

            print(
                f"{event['event_id']} | "
                f"{event['scenario']:<20} | "
                f"{error_type}"
            )


def main():

    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_events.json"
    )

    events = load_events(data_path)

    results = evaluate(events)

    print("=" * 45)
    print("OPSPULSE AI - BASELINE EVALUATION")
    print("=" * 45)

    print(f"Total events:    {results['total']}")
    print(f"True positives:  {results['tp']}")
    print(f"True negatives:  {results['tn']}")
    print(f"False positives: {results['fp']}")
    print(f"False negatives: {results['fn']}")

    print("-" * 45)

    print(f"Precision:       {results['precision']:.2%}")
    print(f"Recall:          {results['recall']:.2%}")
    print(f"Accuracy:        {results['accuracy']:.2%}")
    print(f"F1-score:        {results['f1_score']:.2%}")

    analyze_errors(events)


if __name__ == "__main__":
    main()