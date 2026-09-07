from difflib import SequenceMatcher
from typing import Dict

from utils.normalization import normalize_title


def title_similarity(
    supplied_title: str,
    candidate_title: str,
) -> float:
    """
    Calculate normalized title similarity.

    Returns a value between 0 and 1.
    """

    supplied = normalize_title(supplied_title)
    candidate = normalize_title(candidate_title)

    if not supplied or not candidate:
        return 0.0

    if supplied == candidate:
        return 1.0

    return SequenceMatcher(
        None,
        supplied,
        candidate,
    ).ratio()


def title_match_details(
    supplied_title: str,
    candidate_title: str,
) -> Dict[str, object]:
    """
    Return detailed title comparison information.

    This lets the verifier distinguish:
      - exact match
      - formatting-normalized match
      - substantive difference
    """

    raw_supplied = supplied_title or ""
    raw_candidate = candidate_title or ""

    normalized_supplied = normalize_title(raw_supplied)
    normalized_candidate = normalize_title(raw_candidate)

    if not normalized_supplied or not normalized_candidate:
        return {
            "raw_similarity": 0.0,
            "normalized_similarity": 0.0,
            "match_type": "NO_COMPARISON",
        }

    raw_similarity = SequenceMatcher(
        None,
        raw_supplied.casefold().strip(),
        raw_candidate.casefold().strip(),
    ).ratio()

    normalized_similarity = SequenceMatcher(
        None,
        normalized_supplied,
        normalized_candidate,
    ).ratio()

    if normalized_supplied == normalized_candidate:

        if raw_supplied.strip() == raw_candidate.strip():
            match_type = "EXACT"
        else:
            match_type = "FORMATTING_NORMALIZED"

    elif normalized_similarity >= 0.90:
        match_type = "MINOR_DIFFERENCE"

    else:
        match_type = "SUBSTANTIVE_DIFFERENCE"

    return {
        "raw_similarity": round(raw_similarity, 4),
        "normalized_similarity": round(normalized_similarity, 4),
        "match_type": match_type,
    }