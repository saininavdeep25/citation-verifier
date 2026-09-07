from typing import List

from rapidfuzz import fuzz

from utils.normalization import (
    author_surname,
    normalize_author_name,
)


def _author_match_score(a: str, b: str) -> float:
    """
    Compare two individual authors.

    Examples:
        "He, K." vs "Kaiming He"
        "Vaswani, A." vs "Ashish Vaswani"

    should score very highly.
    """

    surname_a = author_surname(a)
    surname_b = author_surname(b)

    if not surname_a or not surname_b:
        return 0.0

    surname_score = fuzz.ratio(
        surname_a,
        surname_b,
    ) / 100.0

    normalized_a = normalize_author_name(a)
    normalized_b = normalize_author_name(b)

    full_score = fuzz.token_set_ratio(
        normalized_a,
        normalized_b,
    ) / 100.0

    # Surname agreement is the strongest signal.
    return 0.75 * surname_score + 0.25 * full_score


def author_similarity(
    input_authors: List[str],
    candidate_authors: List[str],
) -> float:
    if not input_authors or not candidate_authors:
        return 0.0

    input_authors = [
        a for a in input_authors
        if a and normalize_author_name(a)
    ]

    candidate_authors = [
        a for a in candidate_authors
        if a and normalize_author_name(a)
    ]

    if not input_authors or not candidate_authors:
        return 0.0

    # First author is particularly important.
    first_score = _author_match_score(
        input_authors[0],
        candidate_authors[0],
    )

    # For each cited author, find the best candidate-author match.
    individual_scores = []

    for input_author in input_authors:
        best = max(
            _author_match_score(
                input_author,
                candidate_author,
            )
            for candidate_author in candidate_authors
        )

        individual_scores.append(best)

    average_score = (
        sum(individual_scores)
        / len(individual_scores)
    )

    # If citation uses et al. or only a partial list, do not demand
    # that every database author appear in the citation.
    score = (
        0.55 * first_score
        + 0.45 * average_score
    )

    return min(1.0, score)