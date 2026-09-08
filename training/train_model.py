"""
Train the citation candidate matching model.

Model:
    StandardScaler
    +
    LogisticRegression

The model predicts:

    P(candidate is the correct publication | features)

Features:

    title_similarity
    author_similarity
    venue_similarity
    year_similarity
    doi_similarity
    doi_exact_match
    doi_present
    year_difference
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

TARGET = "label"


def load_data(path):
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.copy()

    df[FEATURES] = df[FEATURES].fillna(0)

    df[TARGET] = df[TARGET].astype(int)

    return df


def train(train_file):
    df = load_data(train_file)

    X = df[FEATURES]
    y = df[TARGET]

    print("Training dataset:")
    print(f"Rows: {len(df)}")
    print(
        f"Positive: {(y == 1).sum()}"
    )
    print(
        f"Negative: {(y == 0).sum()}"
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model, df


def evaluate(
    model,
    df,
    split_name,
):
    X = df[FEATURES]
    y = df[TARGET]

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(
        f"\n===== {split_name} ====="
    )

    print(
        "Accuracy:",
        accuracy_score(y, predictions),
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


def save_model(model, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )

    print(
        f"\nModel saved to: {output_path}"
    )


def save_metadata(
    model,
    output_path,
):
    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = (
        classifier.coef_[0]
    )

    metadata = {
        "model_type": "LogisticRegression",
        "features": FEATURES,
        "coefficients": {
            feature: float(coefficient)
            for feature, coefficient
            in zip(
                FEATURES,
                coefficients,
            )
        },
        "intercept": float(
            classifier.intercept_[0]
        ),
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Metadata saved to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default="data/splits/train.csv",
    )

    parser.add_argument(
        "--validation",
        default="data/splits/validation.csv",
    )

    parser.add_argument(
        "--model",
        default="models/citation_verifier_model.joblib",
    )

    parser.add_argument(
        "--metadata",
        default="models/citation_verifier_model_metadata.json",
    )

    args = parser.parse_args()

    model, train_df = train(
        args.train
    )

    validation_df = load_data(
        args.validation
    )

    evaluate(
        model,
        train_df,
        "TRAIN",
    )

    evaluate(
        model,
        validation_df,
        "VALIDATION",
    )

    save_model(
        model,
        args.model,
    )

    save_metadata(
        model,
        args.metadata,
    )


if __name__ == "__main__":
    main()