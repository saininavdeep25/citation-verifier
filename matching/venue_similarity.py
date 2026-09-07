from rapidfuzz import fuzz
from utils.normalization import normalize_venue


def venue_similarity(a: str, b: str) -> float:
    a = normalize_venue(a)
    b = normalize_venue(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    token = fuzz.token_set_ratio(a, b) / 100.0
    ratio = fuzz.ratio(a, b) / 100.0
    return 0.65 * token + 0.35 * ratio
