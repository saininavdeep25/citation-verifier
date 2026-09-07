import html
import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters and common typography
    differences.
    """

    if not text:
        return ""

    text = html.unescape(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_title(text: Optional[str]) -> str:
    """
    Normalize an academic title for comparison.

    This deliberately removes presentation/extraction artifacts
    but does not attempt to repair arbitrary missing spaces.
    """

    if not text:
        return ""

    text = normalize_unicode(text)

    # Remove HTML tags while preserving their textual content.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Remove common LaTeX formatting commands.
    text = re.sub(
        r"\\textit\{([^{}]*)\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\\textbf\{([^{}]*)\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\\emph\{([^{}]*)\}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining common LaTeX commands.
    text = re.sub(
        r"\\[a-zA-Z]+\{([^{}]*)\}",
        r"\1",
        text,
    )

    # Normalize ampersand to "and".
    text = re.sub(
        r"\s*&\s*",
        " and ",
        text,
    )

    # Normalize hyphen-like characters.
    text = re.sub(
        r"[\u2010\u2011\u2012\u2013\u2014\u2212]",
        "-",
        text,
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # Remove punctuation at the END of a title.
    #
    # This handles extraction differences such as:
    #
    #   Attention Is All You Need
    #   Attention Is All You Need!
    #
    # while preserving punctuation inside the title.
    text = re.sub(
        r"[.!?,;:]+$",
        "",
        text,
    ).strip()

    return text.lower()


def normalize_author_name(name: Optional[str]) -> str:
    """
    Normalize an author's name for comparison.
    """

    if not name:
        return ""

    name = normalize_unicode(name)

    name = name.strip()

    # Remove trailing punctuation.
    name = re.sub(
        r"[.,;:]+$",
        "",
        name,
    )

    # Collapse whitespace.
    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name.lower()


def author_surname(name: Optional[str]) -> str:
    """
    Extract the surname from common citation/database formats.
    """

    if not name:
        return ""

    name = normalize_author_name(name)

    if not name:
        return ""

    # "He, K." -> "he"
    if "," in name:
        return name.split(",", 1)[0].strip()

    # "Kaiming He" -> "he"
    parts = name.split()

    if parts:
        return parts[-1]

    return ""


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """
    Normalize DOI representation.
    """

    if not doi:
        return None

    doi = html.unescape(doi)

    doi = doi.strip()

    doi = re.sub(
        r"^https?://doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    )

    doi = re.sub(
        r"^doi:\s*",
        "",
        doi,
        flags=re.IGNORECASE,
    )

    doi = doi.strip()

    # Remove surrounding brackets/quotes.
    doi = doi.strip(" <>[](){}\"'")

    # Remove trailing citation punctuation.
    doi = doi.rstrip(".,;")

    if not doi:
        return None

    return doi.lower()


def extract_doi(text: Optional[str]) -> Optional[str]:
    """
    Extract a DOI from arbitrary citation text.

    Supports:
        https://doi.org/10.xxxx/xxxxx
        http://doi.org/10.xxxx/xxxxx
        doi:10.xxxx/xxxxx
        10.xxxx/xxxxx
    """

    if not text:
        return None

    text = html.unescape(text)

    pattern = re.compile(
        r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?"
        r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
        re.IGNORECASE,
    )

    match = pattern.search(text)

    if not match:
        return None

    return normalize_doi(
        match.group(1)
    )


VENUE_ALIASES = {
    "neurips": "advances in neural information processing systems",
    "nips": "advances in neural information processing systems",

    "iclr": "international conference on learning representations",

    "cvpr": (
        "proceedings of the ieee conference on "
        "computer vision and pattern recognition"
    ),

    "iccv": (
        "proceedings of the ieee international conference "
        "on computer vision"
    ),

    "eccv": (
        "computer vision – eccv"
    ),

    "acl": (
        "proceedings of the annual meeting of the "
        "association for computational linguistics"
    ),

    "emnlp": (
        "proceedings of the conference on empirical methods "
        "in natural language processing"
    ),

    "naacl": (
        "proceedings of the conference of the north american "
        "chapter of the association for computational linguistics"
    ),

    "aaai": (
        "proceedings of the aaai conference on artificial intelligence"
    ),

    "ijcai": (
        "proceedings of the international joint conference "
        "on artificial intelligence"
    ),

    "sigmod": (
        "proceedings of the acm sigmod international conference "
        "on management of data"
    ),

    "kdd": (
        "proceedings of the acm sigkdd international conference "
        "on knowledge discovery and data mining"
    ),
}



def normalize_venue(text: Optional[str]) -> str:
    """
    Normalize publication venue names and map common aliases
    to canonical venue names.
    """

    if not text:
        return ""

    text = normalize_unicode(text)

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    text = text.lower()

    # Remove common punctuation.
    text = re.sub(r"[.,;:()\[\]{}]", " ", text)

    # Normalize whitespace again after punctuation removal.
    text = re.sub(r"\s+", " ", text).strip()

    
    # Alias → canonical venue.
    return VENUE_ALIASES.get(text, text)