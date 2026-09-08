"""
Download the HALLMARK citation hallucination benchmark.

Usage:
    python training/download_hallmark.py

Requires:
    pip install datasets
"""

from pathlib import Path
import json

from datasets import load_dataset


OUTPUT_DIR = Path("data/hallmark")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HALLMARK dataset...")

    dataset = load_dataset("hallmark-neurips2026/HALLMARK")

    print("\nDataset loaded successfully.")
    print(dataset)

    # Save each split as JSONL
    for split_name, split_dataset in dataset.items():
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"

        print(f"Saving {split_name} -> {output_file}")

        with output_file.open("w", encoding="utf-8") as f:
            for record in split_dataset:
                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    print("\nHALLMARK download complete.")


if __name__ == "__main__":
    main()