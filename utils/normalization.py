import re
import unicodedata
from typing import Iterable, List, Optional


DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?"
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.I,
)


def normalize_unicode(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return (
        value
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""

    value = normalize_unicode(value).lower()

    # Remove URLs because they should not affect title/venue matching.
    value = re.sub(r"https?://\S+", " ", value)

    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"_+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_title(value: Optional[str]) -> str:
    return normalize_text(value)


def normalize_author_name(value: Optional[str]) -> str:
    if not value:
        return ""

    value = normalize_unicode(value).lower().strip()

    # Remove et al. but preserve the actual author.
    value = re.sub(r"\bet\s+al\.?\b", "", value)

    # Normalize punctuation.
    value = re.sub(r"[.,]", " ", value)
    value = re.sub(r"[^\w\s-]", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def author_surname(value: Optional[str]) -> str:
    """
    Extract the surname from common author formats.

    Examples:
        "Vaswani, A."       -> "vaswani"
        "Ashish Vaswani"    -> "vaswani"
        "He, K."            -> "he"
        "Kaiming He"        -> "he"
    """
    if not value:
        return ""

    original = normalize_unicode(value).strip()

    if "," in original:
        surname = original.split(",", 1)[0]
        return normalize_text(surname)

    normalized = normalize_author_name(original)

    if not normalized:
        return ""

    return normalized.split()[-1]


def normalize_authors(authors: Iterable[str]) -> List[str]:
    result = []

    for author in authors:
        normalized = normalize_author_name(author)
        if normalized:
            result.append(normalized)

    return result


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = normalize_unicode(value).strip()

    match = DOI_RE.search(value)

    if match:
        doi = match.group(1)
    else:
        doi = value

        doi = re.sub(
            r"^(?:https?://)?(?:dx\.)?doi\.org/",
            "",
            doi,
            flags=re.I,
        )

        doi = re.sub(
            r"^doi:\s*",
            "",
            doi,
            flags=re.I,
        )

    # Remove citation punctuation.
    doi = doi.rstrip(" \t\r\n.,;:)]}")

    return doi.lower() if doi else None


def extract_doi(text: str) -> Optional[str]:
    if not text:
        return None

    match = DOI_RE.search(text)

    if not match:
        return None

    return normalize_doi(match.group(1))


def normalize_venue(value: Optional[str]) -> str:
    text = normalize_text(value)

    aliases = {
        "neurips": "advances in neural information processing systems",
        "nips": "advances in neural information processing systems",
        "neurips proceedings": "advances in neural information processing systems",
        "acm siggraph": "siggraph",
        "siggraph": "siggraph",
    }

    return aliases.get(text, text)