"""
Split candidate-level data by citation ID.

This prevents leakage between train/validation/test sets.
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_file,
    output_dir,
    train_size=0.70,
    validation_size=0.15,
    test_size=0.15,
    random_state=42,
):
    if abs(
        train_size + validation_size + test_size - 1.0
    ) > 1e-9:
        raise ValueError(
            "Train/validation/test sizes must sum to 1."
        )

    df = pd.read_csv(input_file)

    if "citation_id" not in df.columns:
        raise ValueError(
            "Dataset must contain citation_id."
        )

    citation_ids = (
        df["citation_id"]
        .drop_duplicates()
        .tolist()
    )

    train_ids, temp_ids = train_test_split(
        citation_ids,
        test_size=(validation_size + test_size),
        random_state=random_state,
    )

    relative_test_size = (
        test_size
        / (validation_size + test_size)
    )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test_size,
        random_state=random_state,
    )

    train_df = df[
        df["citation_id"].isin(train_ids)
    ]

    validation_df = df[
        df["citation_id"].isin(validation_ids)
    ]

    test_df = df[
        df["citation_id"].isin(test_ids)
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print("Dataset split complete.")

    print(
        f"Train citations:      {len(train_ids)}"
    )

    print(
        f"Validation citations: {len(validation_ids)}"
    )

    print(
        f"Test citations:       {len(test_ids)}"
    )

    print()

    print(
        f"Train candidate rows:      {len(train_df)}"
    )

    print(
        f"Validation candidate rows: {len(validation_df)}"
    )

    print(
        f"Test candidate rows:       {len(test_df)}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/hallmark_candidates.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="data/splits",
    )

    args = parser.parse_args()

    split_dataset(
        args.input,
        args.output_dir,
    )


if __name__ == "__main__":
    main()