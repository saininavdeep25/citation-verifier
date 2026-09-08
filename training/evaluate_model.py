"""
Evaluate the final citation matching model on the held-out test set.
"""

from pathlib import Path
import argparse

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


FEATURES = [
    "title_similarity",
    "author_similarity",
    "venue_similarity",
    "year_similarity",
    "doi_similarity",
    "doi_exact_match",
    "doi_present",
    "year_difference",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/citation_verifier_model.joblib",
    )

    parser.add_argument(
        "--test",
        default="data/splits/test.csv",
    )

    args = parser.parse_args()

    model = joblib.load(
        args.model
    )

    df = pd.read_csv(
        args.test
    )

    X = df[FEATURES].fillna(0)

    y = df["label"].astype(int)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\n==============================")
    print("FINAL HELD-OUT TEST RESULTS")
    print("==============================\n")

    print(
        "Rows:",
        len(df),
    )

    print(
        "Accuracy:",
        accuracy_score(
            y,
            predictions,
        ),
    )

    print(
        "Precision:",
        precision_score(
            y,
            predictions,
            zero_division=0,
        ),
    )

    print(
        "Recall:",
        recall_score(
            y,
            predictions,
            zero_division=0,
        ),
    )

    print(
        "F1:",
        f1_score(
            y,
            predictions,
            zero_division=0,
        ),
    )

    if len(np.unique(y)) == 2:
        print(
            "ROC-AUC:",
            roc_auc_score(
                y,
                probabilities,
            ),
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()