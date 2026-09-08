"""
Prepare HALLMARK for machine-learning training.

Pipeline:

HALLMARK citation
        |
        v
parse metadata
        |
        v
Crossref/OpenAlex retrieval
        |
        v
candidate publications
        |
        v
feature extraction
        |
        v
candidate-level dataset
        |
        v
training_data.csv

IMPORTANT:
The train/validation/test split is performed at the CITATION level,
not candidate-row level, to prevent leakage.
"""

from pathlib import Path
import argparse
import csv
import time

from training.dataset_utils import (
    load_jsonl,
    extract_fields,
    get_label,
)

from training.build_features import build_feature_vector

from parser.citation_parser import CitationParser

from sources.crossref import CrossrefSource
from sources.openalex import OpenAlexSource


OUTPUT_COLUMNS = [
    "citation_id",
    "citation_label",
    "candidate_source",
    "candidate_id",
    "candidate_title",
    "candidate_authors",
    "candidate_venue",
    "candidate_year",
    "candidate_doi",
    "title_similarity",
    "author_similarity",
    "venue_similarity",
    "year_similarity",
    "doi_similarity",
    "doi_exact_match",
    "doi_present",
    "year_difference",
    "label",
]


def citation_to_parser_input(fields):
    """
    Convert HALLMARK metadata into a citation string.

    This is mainly a fallback representation. If your parser already
    accepts structured metadata, use that directly.
    """

    authors = fields.get("authors") or []

    if isinstance(authors, list):
        author_text = ", ".join(
            str(a) for a in authors
        )
    else:
        author_text = str(authors)

    title = fields.get("title") or ""
    year = fields.get("year") or ""
    venue = fields.get("venue") or ""
    doi = fields.get("doi") or ""

    citation = (
        f"{author_text} "
        f"({year}). "
        f"{title}. "
        f"{venue}. "
        f"{doi}"
    )

    return citation


def normalize_candidate(candidate, source_name):
    """
    Convert a Crossref/OpenAlex candidate to our common schema.
    """

    authors = candidate.get("authors", [])

    if authors is None:
        authors = []

    return {
        "source": source_name,
        "id": candidate.get("id")
        or candidate.get("doi")
        or candidate.get("title"),

        "title": candidate.get("title"),

        "authors": authors,

        "venue": candidate.get("venue"),

        "year": candidate.get("year"),

        "doi": candidate.get("doi"),
    }


def search_candidates(
    citation_metadata,
    crossref,
    openalex,
):
    """
    Retrieve candidates using multiple strategies.

    DOI search is attempted first.
    """

    candidates = []

    title = citation_metadata.get("title")
    authors = citation_metadata.get("authors")
    venue = citation_metadata.get("venue")
    doi = citation_metadata.get("doi")

    # ---------------------------------------------------------
    # 1. DOI search
    # ---------------------------------------------------------

    if doi:
        try:
            result = crossref.search_by_doi(doi)

            if result:
                candidates.append(
                    normalize_candidate(
                        result,
                        "crossref",
                    )
                )
        except Exception as exc:
            print(
                f"Crossref DOI search failed: {exc}"
            )

        try:
            result = openalex.search_by_doi(doi)

            if result:
                candidates.append(
                    normalize_candidate(
                        result,
                        "openalex",
                    )
                )
        except Exception as exc:
            print(
                f"OpenAlex DOI search failed: {exc}"
            )

    # ---------------------------------------------------------
    # 2. Title search
    # ---------------------------------------------------------

    if title:
        try:
            results = crossref.search_by_title(title)

            for result in results:
                candidates.append(
                    normalize_candidate(
                        result,
                        "crossref",
                    )
                )
        except Exception as exc:
            print(
                f"Crossref title search failed: {exc}"
            )

        try:
            results = openalex.search_by_title(title)

            for result in results:
                candidates.append(
                    normalize_candidate(
                        result,
                        "openalex",
                    )
                )
        except Exception as exc:
            print(
                f"OpenAlex title search failed: {exc}"
            )

    # ---------------------------------------------------------
    # 3. Author + title search
    # ---------------------------------------------------------

    if title and authors:
        try:
            first_author = authors[0]

            results = crossref.search_by_author_title(
                first_author,
                title,
            )

            for result in results:
                candidates.append(
                    normalize_candidate(
                        result,
                        "crossref",
                    )
                )
        except Exception as exc:
            print(
                f"Crossref author/title search failed: {exc}"
            )

        try:
            results = openalex.search_by_author_title(
                first_author,
                title,
            )

            for result in results:
                candidates.append(
                    normalize_candidate(
                        result,
                        "openalex",
                    )
                )
        except Exception as exc:
            print(
                f"OpenAlex author/title search failed: {exc}"
            )

    # ---------------------------------------------------------
    # Remove duplicates
    # ---------------------------------------------------------

    unique = {}

    for candidate in candidates:
        key = (
            candidate.get("doi")
            or candidate.get("id")
            or candidate.get("title")
        )

        if key:
            unique[key] = candidate

    return list(unique.values())


def candidate_matches_valid_citation(
    citation_metadata,
    candidate,
):
    """
    Determine whether a candidate is the actual publication
    represented by a VALID HALLMARK citation.

    This uses deterministic identity rules.

    DOI exact match is strongest.

    Otherwise title + author agreement is used.
    """

    citation_doi = citation_metadata.get("doi")
    candidate_doi = candidate.get("doi")

    if citation_doi and candidate_doi:
        from utils.normalization import normalize_doi

        if (
            normalize_doi(citation_doi)
            == normalize_doi(candidate_doi)
        ):
            return True

    features = build_feature_vector(
        citation_metadata,
        candidate,
    )

    title_score = features["title_similarity"]
    author_score = features["author_similarity"]

    # Conservative positive matching rule.
    if (
        title_score >= 0.90
        and author_score >= 0.85
    ):
        return True

    return False


def process_record(
    record,
    crossref,
    openalex,
):
    """
    Process one HALLMARK citation.
    """

    fields = extract_fields(record)

    citation_label = get_label(record)

    if citation_label is None:
        return []

    citation_id = (
        record.get("bibtex_key")
        or record.get("id")
        or record.get("raw_bibtex")
        or str(id(record))
    )

    citation_metadata = {
        "title": fields.get("title"),
        "authors": fields.get("authors") or [],
        "venue": fields.get("venue"),
        "year": fields.get("year"),
        "doi": fields.get("doi"),
    }

    candidates = search_candidates(
        citation_metadata,
        crossref,
        openalex,
    )

    rows = []

    positive_found = False

    for candidate in candidates:

        is_positive = False

        if citation_label == 1:
            is_positive = candidate_matches_valid_citation(
                citation_metadata,
                candidate,
            )

        # For hallucinated citations, candidates retrieved from
        # scholarly databases are treated as negatives unless a
        # strong deterministic match exists.
        elif citation_label == 0:
            is_positive = False

        if is_positive:
            positive_found = True

        features = build_feature_vector(
            citation_metadata,
            candidate,
        )

        row = {
            "citation_id": citation_id,
            "citation_label": citation_label,

            "candidate_source": candidate.get(
                "source"
            ),

            "candidate_id": candidate.get(
                "id"
            ),

            "candidate_title": candidate.get(
                "title"
            ),

            "candidate_authors": str(
                candidate.get("authors")
            ),

            "candidate_venue": candidate.get(
                "venue"
            ),

            "candidate_year": candidate.get(
                "year"
            ),

            "candidate_doi": candidate.get(
                "doi"
            ),

            **features,

            "label": int(is_positive),
        }

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/hallmark/dev_public.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/processed/hallmark_candidates.csv",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    records = load_jsonl(args.input)

    if args.limit:
        records = records[:args.limit]

    print(
        f"Processing {len(records)} HALLMARK citations..."
    )

    crossref = CrossrefSource()
    openalex = OpenAlexSource()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for index, record in enumerate(records, start=1):

        print(
            f"[{index}/{len(records)}] Processing citation..."
        )

        try:
            rows = process_record(
                record,
                crossref,
                openalex,
            )

            all_rows.extend(rows)

        except Exception as exc:
            print(
                f"Error processing citation: {exc}"
            )

        # Avoid hammering APIs.
        time.sleep(0.1)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(all_rows)

    positives = sum(
        1 for row in all_rows
        if row["label"] == 1
    )

    negatives = sum(
        1 for row in all_rows
        if row["label"] == 0
    )

    print("\nDataset preparation complete.")

    print(f"Candidate rows: {len(all_rows)}")
    print(f"Positive rows:   {positives}")
    print(f"Negative rows:   {negatives}")

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()