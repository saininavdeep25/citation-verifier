from difflib import SequenceMatcher
from typing import List

from utils.normalization import (
    author_surname,
    normalize_author_name,
)


def _name_similarity(
    citation_name: str,
    database_name: str,
) -> float:
    """
    Compare two author names.

    Surname agreement is given strong importance because
    academic databases frequently represent authors differently,
    e.g.:

        "He, K."
        "Kaiming He"

    A matching surname should therefore count as a strong match
    even when initials/full names differ.
    """

    citation_normalized = normalize_author_name(citation_name)
    database_normalized = normalize_author_name(database_name)

    if not citation_normalized or not database_normalized:
        return 0.0

    # Exact normalized name
    if citation_normalized == database_normalized:
        return 1.0

    citation_surname = author_surname(citation_name)
    database_surname = author_surname(database_name)

    # Strong match when surnames are identical.
    if (
        citation_surname
        and database_surname
        and citation_surname == database_surname
    ):
        full_name_score = SequenceMatcher(
            None,
            citation_normalized,
            database_normalized,
        ).ratio()

        # Surname is the primary identity signal.
        return (
            0.85 * 1.0
            + 0.15 * full_name_score
        )

    # Different surnames: ordinary fuzzy comparison.
    return SequenceMatcher(
        None,
        citation_normalized,
        database_normalized,
    ).ratio()


def author_similarity(
    citation_authors: List[str],
    database_authors: List[str],
) -> float:
    """
    Calculate similarity between two author lists.

    The first author receives 50% of the total weight because
    first-author identity is particularly informative for
    publication matching.

    Remaining authors share the other 50%.
    """

    if not citation_authors or not database_authors:
        return 0.0

    # Work on copies so the caller's lists are never modified.
    citation = list(citation_authors)
    database = list(database_authors)

    # Match each citation author to the best unused database author.
    used = set()
    scores = []

    for citation_author in citation:

        best_score = 0.0
        best_index = None

        for index, database_author in enumerate(database):

            if index in used:
                continue

            score = _name_similarity(
                citation_author,
                database_author,
            )

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            used.add(best_index)

        scores.append(best_score)

    if not scores:
        return 0.0

    # Single-author citation.
    if len(scores) == 1:
        return scores[0]

    # First author = 50%.
    first_author_score = scores[0]

    # Remaining authors = 50%.
    remaining_scores = scores[1:]
    remaining_average = (
        sum(remaining_scores) / len(remaining_scores)
        if remaining_scores
        else 0.0
    )

    return (
        0.50 * first_author_score
        + 0.50 * remaining_average
    )