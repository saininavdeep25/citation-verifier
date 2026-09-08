from typing import List

from config import (
    TITLE_WEIGHT,
    AUTHOR_WEIGHT,
    VENUE_WEIGHT,
    YEAR_WEIGHT,
    DOI_WEIGHT,
)
from models.schemas import CandidatePaper, CitationMetadata
from matching.title_similarity import (
    title_similarity,
    title_match_details,
)
from matching.author_similarity import author_similarity
from matching.venue_similarity import venue_similarity
from utils.normalization import normalize_doi


def _year_similarity(
    citation_year,
    candidate_year,
) -> float:
    """
    Compare publication years.

    Returns:
        1.0  -> exact match
        0.0  -> mismatch
        0.0  -> unavailable comparison
    """

    if citation_year is None or candidate_year is None:
        return 0.0

    return 1.0 if citation_year == candidate_year else 0.0


def _doi_similarity(
    citation_doi,
    candidate_doi,
) -> float:
    """
    Compare normalized DOIs.

    Returns:
        1.0 -> exact DOI match
        0.0 -> mismatch or unavailable
    """

    input_doi = normalize_doi(citation_doi)
    candidate_doi = normalize_doi(candidate_doi)

    if not input_doi or not candidate_doi:
        return 0.0

    return 1.0 if input_doi == candidate_doi else 0.0


def _weighted_score(
    scores,
    available_fields,
) -> float:
    """
    Calculate a weighted candidate score using only fields
    for which both the citation and candidate contain data.

    This is important because missing metadata should not
    automatically count as negative evidence.
    """

    weights = {
        "title": TITLE_WEIGHT,
        "author": AUTHOR_WEIGHT,
        "venue": VENUE_WEIGHT,
        "year": YEAR_WEIGHT,
        "doi": DOI_WEIGHT,
    }

    numerator = 0.0
    denominator = 0.0

    for field in available_fields:
        weight = weights[field]

        numerator += scores[field] * weight
        denominator += weight

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def _get_available_fields(
    citation: CitationMetadata,
    candidate: CandidatePaper,
) -> List[str]:
    """
    Determine which metadata fields can actually be compared.
    """

    available_fields = []

    if citation.title and candidate.title:
        available_fields.append("title")

    if citation.authors and candidate.authors:
        available_fields.append("author")

    if citation.venue and candidate.venue:
        available_fields.append("venue")

    if (
        citation.year is not None
        and candidate.year is not None
    ):
        available_fields.append("year")

    if citation.doi and candidate.doi:
        available_fields.append("doi")

    return available_fields


def rank_candidate(
    citation: CitationMetadata,
    candidate: CandidatePaper,
) -> CandidatePaper:
    """
    Calculate field-level similarity scores and the final
    candidate score for a single candidate.

    The CandidatePaper object is updated in place and returned.
    """

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if citation.title and candidate.title:

        candidate.title_similarity = title_similarity(
            citation.title,
            candidate.title,
        )

        title_details = title_match_details(
            citation.title,
            candidate.title,
        )

    else:

        candidate.title_similarity = 0.0

        title_details = {
            "raw_similarity": 0.0,
            "normalized_similarity": 0.0,
            "match_type": "NO_COMPARISON",
        }

    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    if citation.authors and candidate.authors:

        candidate.author_similarity = author_similarity(
            citation.authors,
            candidate.authors,
        )

    else:

        candidate.author_similarity = 0.0

    # --------------------------------------------------------
    # Venue
    # --------------------------------------------------------

    if citation.venue and candidate.venue:

        candidate.venue_similarity = venue_similarity(
            citation.venue,
            candidate.venue,
        )

    else:

        candidate.venue_similarity = 0.0

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    year_score = _year_similarity(
        citation.year,
        candidate.year,
    )
    candidate.year_similarity = year_score

    # --------------------------------------------------------
    # DOI
    # --------------------------------------------------------

    doi_score = _doi_similarity(
        citation.doi,
        candidate.doi,
    )
    candidate.doi_similarity = doi_score

    # --------------------------------------------------------
    # Field scores
    # --------------------------------------------------------

    scores = {
        "title": candidate.title_similarity,
        "author": candidate.author_similarity,
        "venue": candidate.venue_similarity,
        "year": year_score,
        "doi": doi_score,
    }

    # --------------------------------------------------------
    # Fields that can actually be compared
    # --------------------------------------------------------

    available_fields = _get_available_fields(
        citation,
        candidate,
    )

    # --------------------------------------------------------
    # Weighted candidate score
    # --------------------------------------------------------

    candidate.candidate_score = _weighted_score(
        scores,
        available_fields,
    )

    # --------------------------------------------------------
    # Preserve title comparison details
    #
    # CandidatePaper may not currently have a dedicated
    # title_match field. Therefore this information is not
    # stored directly on the model here.
    #
    # The classifier independently calculates the same
    # title comparison when making the final decision.
    # --------------------------------------------------------

    return candidate


def rank_candidates(
    citation: CitationMetadata,
    candidates: List[CandidatePaper],
) -> List[CandidatePaper]:
    """
    Rank all candidates for a citation.

    Returns candidates sorted from strongest to weakest.
    """

    ranked_candidates = []

    for candidate in candidates:

        ranked_candidate = rank_candidate(
            citation,
            candidate,
        )

        ranked_candidates.append(
            ranked_candidate
        )

    ranked_candidates.sort(
        key=lambda candidate: candidate.candidate_score,
        reverse=True,
    )

    return ranked_candidates
