"""
Feature extraction for citation-candidate pairs.

The resulting feature vector is:

    title_similarity
    author_similarity
    venue_similarity
    year_similarity
    doi_similarity
    doi_exact_match
    doi_present
    year_difference
"""

from difflib import SequenceMatcher

from matching.title_similarity import title_similarity
from matching.author_similarity import author_similarity
from matching.venue_similarity import venue_similarity
from utils.normalization import normalize_doi


def safe_string(value):
    if value is None:
        return ""

    return str(value).strip()


def calculate_year_similarity(citation_year, candidate_year):
    """
    Calculate year similarity.

    Exact year:
        1.0

    One year difference:
        0.75

    Two years:
        0.50

    Three years:
        0.25

    More than three:
        0.0
    """

    if citation_year is None or candidate_year is None:
        return 0.0

    try:
        y1 = int(citation_year)
        y2 = int(candidate_year)
    except (ValueError, TypeError):
        return 0.0

    difference = abs(y1 - y2)

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.75

    if difference == 2:
        return 0.50

    if difference == 3:
        return 0.25

    return 0.0


def calculate_year_difference(citation_year, candidate_year):
    """
    Absolute difference between publication years.
    """

    if citation_year is None or candidate_year is None:
        return -1

    try:
        return abs(int(citation_year) - int(candidate_year))
    except (ValueError, TypeError):
        return -1


def calculate_doi_features(citation_doi, candidate_doi):
    """
    Calculate DOI-related features.
    """

    citation_doi = normalize_doi(
        safe_string(citation_doi)
    )

    candidate_doi = normalize_doi(
        safe_string(candidate_doi)
    )

    doi_present = 1 if citation_doi else 0

    doi_exact_match = 0

    if citation_doi and candidate_doi:
        if citation_doi == candidate_doi:
            doi_exact_match = 1

    if not citation_doi or not candidate_doi:
        doi_similarity = 0.0
    else:
        doi_similarity = SequenceMatcher(
            None,
            citation_doi.lower(),
            candidate_doi.lower(),
        ).ratio()

    return (
        doi_similarity,
        doi_exact_match,
        doi_present,
    )


def build_feature_vector(citation, candidate):
    """
    Build the complete ML feature vector.

    citation:
        Parsed/generated citation metadata.

    candidate:
        Metadata returned by Crossref/OpenAlex.
    """

    citation_title = safe_string(
        citation.get("title")
    )

    candidate_title = safe_string(
        candidate.get("title")
    )

    citation_authors = citation.get(
        "authors", []
    )

    candidate_authors = candidate.get(
        "authors", []
    )

    citation_venue = safe_string(
        citation.get("venue")
    )

    candidate_venue = safe_string(
        candidate.get("venue")
    )

    citation_year = citation.get("year")
    candidate_year = candidate.get("year")

    title_score = title_similarity(
        citation_title,
        candidate_title,
    )

    author_score = author_similarity(
        citation_authors,
        candidate_authors,
    )

    venue_score = venue_similarity(
        citation_venue,
        candidate_venue,
    )

    year_score = calculate_year_similarity(
        citation_year,
        candidate_year,
    )

    year_difference = calculate_year_difference(
        citation_year,
        candidate_year,
    )

    (
        doi_score,
        doi_exact_match,
        doi_present,
    ) = calculate_doi_features(
        citation.get("doi"),
        candidate.get("doi"),
    )

    return {
        "title_similarity": title_score,
        "author_similarity": author_score,
        "venue_similarity": venue_score,
        "year_similarity": year_score,
        "doi_similarity": doi_score,
        "doi_exact_match": doi_exact_match,
        "doi_present": doi_present,
        "year_difference": year_difference,
    }