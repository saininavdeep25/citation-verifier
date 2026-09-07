from rapidfuzz import fuzz
from utils.normalization import normalize_title


def title_similarity(a: str, b: str) -> float:
    a = normalize_title(a)
    b = normalize_title(b)
    if not a or not b:
        return 0.0

    exact = 1.0 if a == b else 0.0
    ratio = fuzz.token_set_ratio(a, b) / 100.0
    W = fuzz.WRatio(a, b) / 100.0
    return max(exact, 0.65 * ratio + 0.35 * W)
