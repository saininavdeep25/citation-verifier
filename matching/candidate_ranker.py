from typing import List

from config import WEIGHTS
from models.schemas import CandidatePaper, CitationMetadata
from matching.author_similarity import author_similarity
from matching.title_similarity import title_similarity
from matching.venue_similarity import venue_similarity
from utils.normalization import normalize_doi


def year_similarity(
    input_year,
    candidate_year,
):
    if input_year is None or candidate_year is None:
        return None

    try:
        return (
            1.0
            if int(input_year) == int(candidate_year)
            else 0.0
        )
    except (TypeError, ValueError):
        return None


def doi_similarity(
    input_doi,
    candidate_doi,
):
    a = normalize_doi(input_doi)
    b = normalize_doi(candidate_doi)

    if not a or not b:
        return None

    return 1.0 if a == b else 0.0


def score_candidate(
    citation: CitationMetadata,
    candidate: CandidatePaper,
) -> CandidatePaper:

    # -------------------------
    # Individual field scores
    # -------------------------

    candidate.title_similarity = (
        title_similarity(
            citation.title or "",
            candidate.title or "",
        )
    )

    candidate.author_similarity = (
        author_similarity(
            citation.authors,
            candidate.authors,
        )
        if citation.authors and candidate.authors
        else 0.0
    )

    candidate.venue_similarity = (
        venue_similarity(
            citation.venue or "",
            candidate.venue or "",
        )
        if citation.venue and candidate.venue
        else 0.0
    )

    candidate.year_similarity = (
        year_similarity(
            citation.year,
            candidate.year,
        )
        or 0.0
    )

    candidate.doi_similarity = (
        doi_similarity(
            citation.doi,
            candidate.doi,
        )
        or 0.0
    )

    # -------------------------
    # Identity anchor
    # -------------------------

    normalized_input_doi = normalize_doi(
        citation.doi
    )

    normalized_candidate_doi = normalize_doi(
        candidate.doi
    )

    # Exact DOI is the strongest possible identity signal.
    if (
        normalized_input_doi
        and normalized_candidate_doi
        and normalized_input_doi
        == normalized_candidate_doi
    ):
        candidate.candidate_score = 1.0
        return candidate

    # -------------------------
    # Weighted score
    # -------------------------

    available_fields = []

    if citation.title and candidate.title:
        available_fields.append(
            ("title", candidate.title_similarity)
        )

    if citation.authors and candidate.authors:
        available_fields.append(
            ("author", candidate.author_similarity)
        )

    if citation.venue and candidate.venue:
        available_fields.append(
            ("venue", candidate.venue_similarity)
        )

    if citation.year is not None and candidate.year is not None:
        available_fields.append(
            ("year", candidate.year_similarity)
        )

    if citation.doi and candidate.doi:
        available_fields.append(
            ("doi", candidate.doi_similarity)
        )

    if not available_fields:
        candidate.candidate_score = 0.0
        return candidate

    total_weight = sum(
        WEIGHTS[field]
        for field, _ in available_fields
    )

    weighted_score = sum(
        WEIGHTS[field] * score
        for field, score in available_fields
    )

    # Normalize over fields that actually exist.
    candidate.candidate_score = (
        weighted_score / total_weight
        if total_weight > 0
        else 0.0
    )

    return candidate


def rank_candidates(
    citation: CitationMetadata,
    candidates: List[CandidatePaper],
) -> List[CandidatePaper]:

    scored = [
        score_candidate(citation, candidate)
        for candidate in candidates
    ]

    return sorted(
        scored,
        key=lambda c: c.candidate_score,
        reverse=True,
    )