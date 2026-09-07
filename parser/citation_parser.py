import re
from typing import List, Optional

from models.schemas import CitationMetadata
from utils.normalization import extract_doi, normalize_doi


YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")

URL_RE = re.compile(
    r"https?://[^\s\]\[<>()]+",
    re.I,
)


def _clean(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\r\n")

    return value or None


def _extract_year(text: str) -> Optional[int]:
    if not text:
        return None

    # Prefer "(2017)" / "(2017a)"
    match = re.search(
        r"\(\s*((?:18|19|20)\d{2})[a-z]?\s*\)",
        text,
        re.I,
    )

    if match:
        return int(match.group(1))

    # Otherwise use first plausible year.
    match = YEAR_RE.search(text)

    if match:
        return int(match.group(0))

    return None


def _remove_urls_and_doi(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text, flags=re.I)

    text = re.sub(
        r"(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        "",
        text,
        flags=re.I,
    )

    return text


def _split_apa_authors(author_text: str) -> List[str]:
    """
    Parse APA-style authors.

    Example:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
        Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I.

    into:

        [
            "Vaswani, A.",
            "Shazeer, N.",
            ...
        ]
    """
    if not author_text:
        return []

    author_text = author_text.strip()

    author_text = re.sub(
        r"\bet\s+al\.?\s*$",
        "",
        author_text,
        flags=re.I,
    ).strip()

    # Normalize "&" to comma.
    author_text = re.sub(r"\s*&\s*", ", ", author_text)

    # APA author boundaries are generally comma followed by a surname,
    # initials, and another comma OR end of string.
    pattern = re.compile(
        r"""
        (
            [A-Za-zÀ-ÖØ-öø-ÿ'`-]+
            (?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'`-]+)*
            ,
            \s*
            [A-Za-z](?:\.[A-Za-z])*(?:\.\s*)?
            (?:[A-Za-z]\.)?
            (?:\s+[A-Za-z](?:\.[A-Za-z])*)?
        )
        (?=\s*,|\s*$)
        """,
        re.VERBOSE,
    )

    matches = pattern.findall(author_text)

    if matches:
        return [
            _clean(author)
            for author in matches
            if _clean(author)
        ][:20]

    # Fallback for simpler forms.
    parts = re.split(r"\s*;\s*", author_text)

    if len(parts) > 1:
        return [
            _clean(p)
            for p in parts
            if _clean(p)
        ][:20]

    return [_clean(author_text)] if _clean(author_text) else []


def _split_non_apa_authors(author_text: str) -> List[str]:
    if not author_text:
        return []

    author_text = author_text.strip(" ,.;")

    if not author_text:
        return []

    author_text = re.sub(
        r"\bet\s+al\.?\s*$",
        "",
        author_text,
        flags=re.I,
    )

    # Semicolon separated references.
    if ";" in author_text:
        parts = re.split(r"\s*;\s*", author_text)
        return [
            _clean(p)
            for p in parts
            if _clean(p)
        ][:20]

    # "and" / "&" separated.
    if re.search(r"\s+(?:and|&)\s+", author_text, re.I):
        parts = re.split(
            r"\s+(?:and|&)\s+",
            author_text,
            flags=re.I,
        )

        return [
            _clean(p)
            for p in parts
            if _clean(p)
        ][:20]

    # IEEE style often looks like:
    # A. Vaswani, N. Shazeer, N. Parmar
    #
    # Split on commas only when another author-like initial/name follows.
    parts = re.split(
        r",\s*(?=(?:[A-Z]\.\s*)?[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+)*\b)",
        author_text,
    )

    return [
        _clean(p)
        for p in parts
        if _clean(p)
    ][:20]


def _extract_authors(text: str, year: Optional[int]) -> List[str]:
    if not text:
        return []

    work = text.strip()

    if year is not None:
        year_match = re.search(
            rf"\(?\s*{year}[a-z]?\s*\)?",
            work,
            re.I,
        )

        if year_match:
            prefix = work[:year_match.start()]
        else:
            prefix = work
    else:
        prefix = work

    prefix = prefix.strip(" .,:;[]()")

    prefix = re.sub(
        r"^\s*(?:authors?|by)\s*:\s*",
        "",
        prefix,
        flags=re.I,
    )

    prefix = re.sub(
        r"^\s*(?:\[\d+\]|\d+\.)\s*",
        "",
        prefix,
    )

    if not prefix:
        return []

    # If a quoted title exists, everything before it is the author section.
    quote_match = re.search(r'["“]', prefix)

    if quote_match:
        prefix = prefix[:quote_match.start()].strip()

    # APA is the dominant author-year pattern.
    if re.search(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`-]+,\s*[A-Z]", prefix):
        authors = _split_apa_authors(prefix)

        if authors:
            return authors

    return _split_non_apa_authors(prefix)


def _extract_title_and_venue(
    text: str,
    year: Optional[int],
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract title and venue from common author-year-title-venue citations.

    Example:
        (2017). Attention Is All You Need.
        Advances in Neural Information Processing Systems, 30.

    -> title:
        Attention Is All You Need

    -> venue:
        Advances in Neural Information Processing Systems
    """
    work = _remove_urls_and_doi(text).strip()

    if year is not None:
        year_match = re.search(
            rf"\(?\s*{year}[a-z]?\s*\)?",
            work,
            re.I,
        )

        if year_match:
            after_year = work[year_match.end():]
        else:
            after_year = work
    else:
        after_year = work

    after_year = after_year.strip(" .,:;[]()")

    # Quoted title is the strongest title signal.
    quoted = re.search(
        r'"([^"]{3,300})"|“([^”]{3,300})”',
        after_year,
    )

    if quoted:
        title = _clean(quoted.group(1) or quoted.group(2))

        if not title:
            return None, None

        after_title = after_year[quoted.end():]
        venue = _clean_venue(after_title)

        return title, venue

    # Split the post-year portion at the first sentence boundary.
    #
    # This is the critical fix for:
    # "Attention Is All You Need. Advances in Neural Information Processing Systems, 30."
    #
    sentence_parts = re.split(
        r"\.\s+(?=[A-Z0-9])",
        after_year,
        maxsplit=1,
    )

    if sentence_parts:
        title = _clean(sentence_parts[0])

        if title:
            venue = (
                _clean_venue(sentence_parts[1])
                if len(sentence_parts) > 1
                else None
            )

            # Remove obvious bibliographic noise accidentally attached to title.
            title = re.sub(
                r"\s*,?\s*(?:vol(?:ume)?\.?\s*\d+).*$",
                "",
                title,
                flags=re.I,
            )

            title = re.sub(
                r"\s*,?\s*(?:pp?\.?|pages?)\s*\d+.*$",
                "",
                title,
                flags=re.I,
            )

            title = _clean(title)

            return title, venue

    return None, None


def _clean_venue(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    venue = value.strip(" .,:;[]()")

    # Stop at volume/issue/pages information.
    venue = re.split(
        r"\b(?:vol(?:ume)?\.?|no\.?|issue|pp?\.?|pages?)\b",
        venue,
        maxsplit=1,
        flags=re.I,
    )[0]

    # Remove trailing volume numbers such as:
    # "Advances in Neural Information Processing Systems, 30"
    venue = re.sub(
        r",\s*\d+\s*$",
        "",
        venue,
    )

    venue = _clean(venue)

    if not venue or len(venue.split()) < 2:
        return None

    return venue


def _extract_venue(
    text: str,
    title: Optional[str],
) -> Optional[str]:
    if not title:
        return None

    idx = text.lower().find(title.lower())

    if idx < 0:
        return None

    after = text[idx + len(title):]

    return _clean_venue(after)


def parse_citation(citation: str) -> CitationMetadata:
    citation = (citation or "").strip()

    if not citation:
        return CitationMetadata()

    doi = extract_doi(citation)
    year = _extract_year(citation)

    authors = _extract_authors(citation, year)

    title, venue = _extract_title_and_venue(
        citation,
        year,
    )

    # Fallback venue extraction.
    if not venue:
        venue = _extract_venue(
            _remove_urls_and_doi(citation),
            title,
        )

    url_match = URL_RE.search(citation)

    url = (
        url_match.group(0).rstrip(".,);")
        if url_match
        else None
    )

    volume = None
    issue = None
    pages = None

    vm = re.search(
        r"\b(?:vol(?:ume)?\.?\s*)(\d+)",
        citation,
        re.I,
    )

    im = re.search(
        r"\b(?:no\.?|issue)\s*([A-Za-z0-9-]+)",
        citation,
        re.I,
    )

    pm = re.search(
        r"\b(?:pp?\.?|pages?)\s*"
        r"([0-9]+(?:\s*[-–]\s*[0-9]+)?)",
        citation,
        re.I,
    )

    if vm:
        volume = vm.group(1)

    if im:
        issue = im.group(1)

    if pm:
        pages = pm.group(1)

    return CitationMetadata(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=normalize_doi(doi),
        volume=volume,
        issue=issue,
        pages=pages,
        url=url,
    )