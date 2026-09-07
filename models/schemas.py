from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CitationMetadata:
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePaper:
    source: str
    source_id: Optional[str]
    title: Optional[str]
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    url: Optional[str]
    retrieval_method: str
    query: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    # Filled by the ranker
    title_similarity: float = 0.0
    author_similarity: float = 0.0
    venue_similarity: float = 0.0
    year_similarity: float = 0.0
    doi_similarity: float = 0.0
    candidate_score: float = 0.0
    retrieval_methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Raw API payloads can be very large; keep them out of the normal result.
        data.pop("raw", None)
        return data
