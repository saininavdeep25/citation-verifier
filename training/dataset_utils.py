"""
Utility functions for working with HALLMARK records.
"""

from pathlib import Path
import json


def load_jsonl(path):
    """
    Load a JSONL file.

    Returns:
        list[dict]
    """
    path = Path(path)

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: Could not parse line {line_number}: {exc}"
                )

    return records


def save_jsonl(records, path):
    """
    Save records as JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def extract_fields(record):
    """
    Extract normalized citation metadata from a HALLMARK record.

    HALLMARK stores bibliographic information inside the `fields`
    object.
    """

    fields = record.get("fields", {})

    if fields is None:
        fields = {}

    return {
        "title": fields.get("title"),
        "authors": fields.get("author"),
        "year": fields.get("year"),
        "doi": fields.get("doi"),
        "venue": fields.get("venue"),
        "pages": fields.get("pages"),
        "volume": fields.get("volume"),
        "issue": fields.get("issue"),
    }


def get_label(record):
    """
    Convert HALLMARK label to binary label.

    VALID -> 1
    HALLUCINATED -> 0
    """

    label = str(record.get("label", "")).upper().strip()

    if label == "VALID":
        return 1

    if label == "HALLUCINATED":
        return 0

    return None