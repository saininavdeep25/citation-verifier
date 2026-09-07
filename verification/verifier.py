import logging
from typing import Dict, List, Tuple

from models.schemas import (
    CandidatePaper,
    CitationMetadata,
)
from sources.base import ScholarlySource
from matching.candidate_ranker import rank_candidates
from verification.classifier import classify
from utils.normalization import (
    normalize_doi,
    normalize_title,
)


logger = logging.getLogger(
    "citation_verifier"
)


class CitationVerifier:

    def __init__(
        self,
        sources: List[ScholarlySource],
    ):
        self.sources = sources

    @staticmethod
    def _candidate_key(
        candidate: CandidatePaper,
    ) -> str:

        if candidate.doi:
            return (
                "doi:"
                + normalize_doi(candidate.doi)
            )

        title = normalize_title(
            candidate.title
        )

        first_author = ""

        if candidate.authors:
            first_author = normalize_title(
                candidate.authors[0]
            )

        return (
            f"title-author:"
            f"{title}|{first_author}"
        )

    def _record_evidence(
        self,
        evidence: List[Dict],
        source: ScholarlySource,
        method: str,
        query: str,
        candidates_returned: int,
    ):
        evidence.append({
            "method": method,
            "source": source.name,
            "query": query,
            "candidates_returned": candidates_returned,
            "success": (
                source.last_error is None
            ),
            "status_code": (
                getattr(
                    source,
                    "last_status_code",
                    None,
                )
            ),
            "error": (
                getattr(
                    source,
                    "last_error",
                    None,
                )
            ),
        })

    def retrieve(
        self,
        citation: CitationMetadata,
    ) -> Tuple[
        List[CandidatePaper],
        List[Dict],
        bool,
    ]:

        all_candidates: List[
            CandidatePaper
        ] = []

        evidence: List[Dict] = []

        any_success = False
        any_failure = False

        for source in self.sources:

            # ==================================================
            # DOI lookup
            # ==================================================

            if citation.doi:

                candidate = source.lookup_doi(
                    citation.doi
                )

                if getattr(
                    source,
                    "last_error",
                    None,
                ):
                    any_failure = True
                else:
                    any_success = True

                self._record_evidence(
                    evidence,
                    source,
                    "doi",
                    citation.doi,
                    1 if candidate else 0,
                )

                if candidate:
                    all_candidates.append(
                        candidate
                    )

            # ==================================================
            # Title search
            # ==================================================

            if citation.title:

                candidates = (
                    source.search_title(
                        citation.title
                    )
                )

                if getattr(
                    source,
                    "last_error",
                    None,
                ):
                    any_failure = True
                else:
                    any_success = True

                self._record_evidence(
                    evidence,
                    source,
                    "title",
                    citation.title,
                    len(candidates),
                )

                all_candidates.extend(
                    candidates
                )

                # ==================================================
                # Author + title
                # ==================================================

                first_author = (
                    citation.authors[0]
                    if citation.authors
                    else ""
                )

                if first_author:

                    candidates = (
                        source.search_author_title(
                            first_author,
                            citation.title,
                        )
                    )

                    if getattr(
                        source,
                        "last_error",
                        None,
                    ):
                        any_failure = True
                    else:
                        any_success = True

                    self._record_evidence(
                        evidence,
                        source,
                        "author_title",
                        (
                            f"{first_author} + "
                            f"{citation.title}"
                        ),
                        len(candidates),
                    )

                    all_candidates.extend(
                        candidates
                    )

                # ==================================================
                # Title + venue
                # ==================================================

                if citation.venue:

                    candidates = (
                        source.search_title_venue(
                            citation.title,
                            citation.venue,
                        )
                    )

                    if getattr(
                        source,
                        "last_error",
                        None,
                    ):
                        any_failure = True
                    else:
                        any_success = True

                    self._record_evidence(
                        evidence,
                        source,
                        "title_venue",
                        (
                            f"{citation.title} + "
                            f"{citation.venue}"
                        ),
                        len(candidates),
                    )

                    all_candidates.extend(
                        candidates
                    )

        # ======================================================
        # Merge duplicate records
        # ======================================================

        merged: Dict[
            str,
            CandidatePaper,
        ] = {}

        for candidate in all_candidates:

            key = self._candidate_key(
                candidate
            )

            if key not in merged:

                merged[key] = candidate

            else:

                existing = merged[key]

                existing.retrieval_methods = sorted(
                    set(
                        existing.retrieval_methods
                        + candidate.retrieval_methods
                        + [
                            candidate.retrieval_method
                        ]
                    )
                )

                # Prefer richer metadata.
                if (
                    not existing.title
                    and candidate.title
                ):
                    existing.title = (
                        candidate.title
                    )

                if (
                    not existing.authors
                    and candidate.authors
                ):
                    existing.authors = (
                        candidate.authors
                    )

                if (
                    not existing.year
                    and candidate.year
                ):
                    existing.year = (
                        candidate.year
                    )

                if (
                    not existing.venue
                    and candidate.venue
                ):
                    existing.venue = (
                        candidate.venue
                    )

                if (
                    not existing.doi
                    and candidate.doi
                ):
                    existing.doi = (
                        candidate.doi
                    )

                if (
                    not existing.url
                    and candidate.url
                ):
                    existing.url = (
                        candidate.url
                    )

                if (
                    not existing.volume
                    and candidate.volume
                ):
                    existing.volume = (
                        candidate.volume
                    )

                if (
                    not existing.issue
                    and candidate.issue
                ):
                    existing.issue = (
                        candidate.issue
                    )

                if (
                    not existing.pages
                    and candidate.pages
                ):
                    existing.pages = (
                        candidate.pages
                    )

        source_failure = (
            any_failure
            and not any_success
        )

        return (
            list(merged.values()),
            evidence,
            source_failure,
        )

    def verify(
        self,
        citation: CitationMetadata,
    ) -> Dict:

        (
            candidates,
            retrieval_evidence,
            source_failure,
        ) = self.retrieve(citation)

        ranked = rank_candidates(
            citation,
            candidates,
        )

        best = (
            ranked[0]
            if ranked
            else None
        )

        second = (
            ranked[1]
            if len(ranked) > 1
            else None
        )

        classification = classify(
            citation,
            best,
            second_candidate=second,
            source_failure=source_failure,
        )

        field_comparison = None

        if best:

            field_comparison = {
                "title": {
                    "input": citation.title,
                    "database": best.title,
                    "similarity": (
                        best.title_similarity
                    ),
                    "match": (
                        best.title_similarity >= 0.90
                        if citation.title
                        and best.title
                        else None
                    ),
                },

                "authors": {
                    "input": citation.authors,
                    "database": best.authors,
                    "similarity": (
                        best.author_similarity
                    ),
                    "match": (
                        best.author_similarity >= 0.80
                        if citation.authors
                        and best.authors
                        else None
                    ),
                },

                "year": {
                    "input": citation.year,
                    "database": best.year,
                    "match": (
                        citation.year
                        == best.year
                        if (
                            citation.year
                            is not None
                            and best.year
                            is not None
                        )
                        else None
                    ),
                },

                "venue": {
                    "input": citation.venue,
                    "database": best.venue,
                    "similarity": (
                        best.venue_similarity
                    ),
                    "match": (
                        best.venue_similarity >= 0.70
                        if citation.venue
                        and best.venue
                        else None
                    ),
                },

                "doi": {
                    "input": citation.doi,
                    "database": best.doi,
                    "match": (
                        normalize_doi(
                            citation.doi
                        )
                        == normalize_doi(
                            best.doi
                        )
                        if citation.doi
                        and best.doi
                        else None
                    ),
                },
            }

        return {
            "parsed_metadata": (
                citation.to_dict()
            ),

            "status": (
                classification["status"]
            ),

            "confidence": round(
                best.candidate_score,
                4,
            ) if best else 0.0,

            "best_match": (
                best.to_dict()
                if best
                else None
            ),

            "field_comparison": (
                field_comparison
            ),

            "retrieval_evidence": (
                retrieval_evidence
            ),

            "ranked_candidates": [
                c.to_dict()
                for c in ranked[:10]
            ],

            "explanation": (
                classification["reason"]
            ),
        }