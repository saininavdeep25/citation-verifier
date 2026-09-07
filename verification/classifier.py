from typing import Dict, Optional

from config import (
    AMBIGUITY_MARGIN,
    STRONG_AUTHOR_THRESHOLD,
    STRONG_TITLE_ONLY_THRESHOLD,
    STRONG_TITLE_THRESHOLD,
)
from models.schemas import CandidatePaper, CitationMetadata
from utils.normalization import normalize_doi


def _metadata_errors(
    citation: CitationMetadata,
    candidate: CandidatePaper,
):
    errors = []

    # Title
    if (
        citation.title
        and candidate.title
        and candidate.title_similarity < 0.90
    ):
        errors.append("title")

    # Authors
    if (
        citation.authors
        and candidate.authors
        and candidate.author_similarity < STRONG_AUTHOR_THRESHOLD
    ):
        errors.append("authors")

    # Year
    if (
        citation.year is not None
        and candidate.year is not None
        and citation.year != candidate.year
    ):
        errors.append("year")

    # Venue
    if (
        citation.venue
        and candidate.venue
        and candidate.venue_similarity < 0.70
    ):
        errors.append("venue")

    return errors


def _is_exact_doi_match(
    citation: CitationMetadata,
    candidate: CandidatePaper,
) -> bool:

    input_doi = normalize_doi(citation.doi)
    candidate_doi = normalize_doi(candidate.doi)

    return bool(
        input_doi
        and candidate_doi
        and input_doi == candidate_doi
    )


def _is_strong_title_author_match(
    candidate: CandidatePaper,
) -> bool:

    return (
        candidate.title_similarity
        >= STRONG_TITLE_THRESHOLD
        and candidate.author_similarity
        >= STRONG_AUTHOR_THRESHOLD
    )


def _is_strong_title_match(
    candidate: CandidatePaper,
) -> bool:

    return (
        candidate.title_similarity
        >= STRONG_TITLE_ONLY_THRESHOLD
    )


def classify(
    citation: CitationMetadata,
    candidate: Optional[CandidatePaper],
    second_candidate: Optional[CandidatePaper] = None,
    source_failure: bool = False,
) -> Dict[str, str]:

    # ========================================================
    # No candidate
    # ========================================================

    if candidate is None:

        if source_failure:
            return {
                "status": "UNCERTAIN",
                "reason": (
                    "The scholarly databases could not be queried "
                    "successfully, so the citation could not be "
                    "reliably verified."
                ),
            }

        return {
            "status": "LIKELY_HALLUCINATED",
            "reason": (
                "No sufficiently reliable corresponding publication "
                "was found in the searched scholarly databases."
            ),
        }

    # ========================================================
    # Exact DOI identity anchor
    # ========================================================

    if _is_exact_doi_match(
        citation,
        candidate,
    ):

        metadata_errors = _metadata_errors(
            citation,
            candidate,
        )

        if metadata_errors:
            return {
                "status": "EXISTS_METADATA_ERROR",
                "reason": (
                    "The supplied DOI identifies a real publication, "
                    "but one or more supplied bibliographic fields "
                    "do not match the publication metadata: "
                    + ", ".join(metadata_errors)
                    + "."
                ),
            }

        return {
            "status": "VERIFIED",
            "reason": (
                "The supplied DOI exactly identifies the matched "
                "publication, and the supplied bibliographic metadata "
                "is consistent with the database metadata."
            ),
        }

    # ========================================================
    # Strong title + author identity
    # ========================================================

    strong_title_author = (
        _is_strong_title_author_match(candidate)
    )

    # ========================================================
    # Strong title-only identity
    # ========================================================

    strong_title = _is_strong_title_match(candidate)

    # ========================================================
    # Ambiguity check
    # ========================================================

    if (
        strong_title
        and second_candidate is not None
        and not _is_exact_doi_match(
            citation,
            second_candidate,
        )
    ):

        difference = (
            candidate.candidate_score
            - second_candidate.candidate_score
        )

        # If another candidate is nearly as strong, don't
        # arbitrarily declare identity.
        if difference < AMBIGUITY_MARGIN:
            return {
                "status": "UNCERTAIN",
                "reason": (
                    "Multiple plausible publications were found, "
                    "but the evidence does not distinguish the "
                    "best candidate with sufficient confidence."
                ),
            }

    # ========================================================
    # Strong title + author
    # ========================================================

    if strong_title_author:

        # Supplied DOI exists but points somewhere else.
        if (
            citation.doi
            and candidate.doi
            and normalize_doi(citation.doi)
            != normalize_doi(candidate.doi)
        ):
            return {
                "status": "EXISTS_DOI_ERROR",
                "reason": (
                    "A strong corresponding publication was found "
                    "based on title and authors, but the supplied DOI "
                    "does not match the DOI of that publication."
                ),
            }

        metadata_errors = _metadata_errors(
            citation,
            candidate,
        )

        if metadata_errors:
            return {
                "status": "EXISTS_METADATA_ERROR",
                "reason": (
                    "A corresponding publication was identified "
                    "from the bibliographic metadata, but one or "
                    "more supplied fields differ from the database "
                    "metadata: "
                    + ", ".join(metadata_errors)
                    + "."
                ),
            }

        return {
            "status": "VERIFIED",
            "reason": (
                "A strong corresponding publication was identified "
                "using the title and author information, and the "
                "supplied bibliographic metadata is consistent with "
                "the database metadata."
            ),
        }

    # ========================================================
    # Strong title only
    # ========================================================

    if strong_title:

        # Wrong DOI but very strong title.
        if (
            citation.doi
            and candidate.doi
            and normalize_doi(citation.doi)
            != normalize_doi(candidate.doi)
        ):
            return {
                "status": "EXISTS_DOI_ERROR",
                "reason": (
                    "A publication with a strongly matching title "
                    "was found, but the supplied DOI does not match "
                    "the DOI of the identified publication."
                ),
            }

        metadata_errors = _metadata_errors(
            citation,
            candidate,
        )

        if metadata_errors:
            return {
                "status": "EXISTS_METADATA_ERROR",
                "reason": (
                    "A publication with a strongly matching title "
                    "was found, but one or more supplied metadata "
                    "fields differ from the database metadata: "
                    + ", ".join(metadata_errors)
                    + "."
                ),
            }

        return {
            "status": "VERIFIED",
            "reason": (
                "A publication with a strongly matching title was "
                "found and the available bibliographic metadata is "
                "consistent with the database record."
            ),
        }

    # ========================================================
    # Moderate candidate
    # ========================================================

    if candidate.candidate_score >= 0.60:
        return {
            "status": "UNCERTAIN",
            "reason": (
                "One or more plausible candidates were found, but "
                "the available bibliographic evidence is insufficient "
                "for a reliable publication identity match."
            ),
        }

    # ========================================================
    # Weak candidate
    # ========================================================

    if source_failure:
        return {
            "status": "UNCERTAIN",
            "reason": (
                "The available search evidence is weak and one or "
                "more scholarly database requests failed, so the "
                "citation cannot be reliably classified."
            ),
        }

    return {
        "status": "LIKELY_HALLUCINATED",
        "reason": (
            "No sufficiently reliable corresponding publication "
            "was found in the searched scholarly databases."
        ),
    }