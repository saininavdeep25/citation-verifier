import os


# ============================================================
# API configuration
# ============================================================

CROSSREF_BASE_URL = "https://api.crossref.org/v1"
OPENALEX_BASE_URL = "https://api.openalex.org"

CROSSREF_EMAIL = os.getenv(
    "CROSSREF_EMAIL",
    "",
)

OPENALEX_EMAIL = os.getenv(
    "OPENALEX_EMAIL",
    "" if not os.getenv("OPENALEX_EMAIL") else os.getenv("OPENALEX_EMAIL"),
)

REQUEST_TIMEOUT = float(
    os.getenv("REQUEST_TIMEOUT", "15")
)

MAX_CANDIDATES_PER_QUERY = int(
    os.getenv("MAX_CANDIDATES_PER_QUERY", "10")
)


# ============================================================
# Candidate ranking weights
# ============================================================

# Candidate ranking weights
TITLE_WEIGHT = 0.45
AUTHOR_WEIGHT = 0.30
VENUE_WEIGHT = 0.10
YEAR_WEIGHT = 0.05
DOI_WEIGHT = 0.10

WEIGHTS = {
    "title": 0.45,
    "author": 0.30,
    "venue": 0.10,
    "year": 0.05,
    "doi": 0.10,
}


# ============================================================
# Verification thresholds
# ============================================================

# Strong identity match based on title + authors.
STRONG_TITLE_THRESHOLD = float(
    os.getenv(
        "STRONG_TITLE_THRESHOLD",
        "0.90",
    )
)

STRONG_AUTHOR_THRESHOLD = float(
    os.getenv(
        "STRONG_AUTHOR_THRESHOLD",
        "0.80",
    )
)

# Strong title alone.
STRONG_TITLE_ONLY_THRESHOLD = float(
    os.getenv(
        "STRONG_TITLE_ONLY_THRESHOLD",
        "0.95",
    )
)

# Used when deciding whether the top candidate is clearly
# better than the second-best candidate.
AMBIGUITY_MARGIN = float(
    os.getenv(
        "AMBIGUITY_MARGIN",
        "0.05",
    )
)

UNCERTAIN_THRESHOLD = float(
    os.getenv(
        "UNCERTAIN_THRESHOLD",
        "0.60",
    )
)


# ============================================================
# API retry configuration
# ============================================================

MAX_RETRIES = int(
    os.getenv(
        "MAX_RETRIES",
        "4",
    )
)

BACKOFF_BASE_SECONDS = float(
    os.getenv(
        "BACKOFF_BASE_SECONDS",
        "2",
    )
)

MAX_BACKOFF_SECONDS = float(
    os.getenv(
        "MAX_BACKOFF_SECONDS",
        "30",
    )
)


# ============================================================
# HTTP
# ============================================================

USER_AGENT = (
    "AcademicCitationVerifier/1.0 "
    "(research project; citation verification)"
)
