import logging
import random
import time
from typing import Any, Dict, List, Optional

import requests

from config import (
    BACKOFF_BASE_SECONDS,
    MAX_BACKOFF_SECONDS,
    MAX_CANDIDATES_PER_QUERY,
    MAX_RETRIES,
    OPENALEX_BASE_URL,
    OPENALEX_EMAIL,
    REQUEST_TIMEOUT,
    USER_AGENT,
)
from models.schemas import CandidatePaper
from sources.base import ScholarlySource
from utils.normalization import normalize_doi


logger = logging.getLogger("citation_verifier")


class OpenAlexSource(ScholarlySource):
    name = "OpenAlex"

    RETRYABLE_STATUS_CODES = {
        429,
        500,
        502,
        503,
        504,
    }

    def __init__(self):
        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": USER_AGENT,
        })

        self.last_error: Optional[str] = None
        self.last_status_code: Optional[int] = None

    def _reset_error(self):
        self.last_error = None
        self.last_status_code = None

    def _get(
        self,
        path: str = "/works",
        params: Optional[Dict[str, Any]] = None,
    ):

        self._reset_error()

        params = dict(params or {})

        if OPENALEX_EMAIL:
            params["mailto"] = OPENALEX_EMAIL

        for attempt in range(MAX_RETRIES):

            try:
                response = self.session.get(
                    f"{OPENALEX_BASE_URL}{path}",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                self.last_status_code = response.status_code

                if response.status_code == 200:
                    return response.json()

                if response.status_code in self.RETRYABLE_STATUS_CODES:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = (
                                BACKOFF_BASE_SECONDS
                                * (2 ** attempt)
                            )
                    else:
                        delay = (
                            BACKOFF_BASE_SECONDS
                            * (2 ** attempt)
                        )

                    delay = min(
                        delay,
                        MAX_BACKOFF_SECONDS,
                    )

                    delay += random.uniform(
                        0,
                        0.5,
                    )

                    logger.warning(
                        "OpenAlex returned HTTP %s. "
                        "Retrying in %.2f seconds "
                        "(attempt %s/%s).",
                        response.status_code,
                        delay,
                        attempt + 1,
                        MAX_RETRIES,
                    )

                    time.sleep(delay)
                    continue

                response.raise_for_status()

            except requests.RequestException as exc:

                self.last_error = str(exc)

                logger.error(
                    "OpenAlex request failed: %s",
                    exc,
                )

                if attempt < MAX_RETRIES - 1:
                    delay = min(
                        BACKOFF_BASE_SECONDS
                        * (2 ** attempt),
                        MAX_BACKOFF_SECONDS,
                    )

                    time.sleep(delay)
                    continue

                return None

            except ValueError as exc:

                self.last_error = (
                    "Invalid JSON returned by OpenAlex: "
                    + str(exc)
                )

                logger.error(
                    "OpenAlex returned invalid JSON: %s",
                    exc,
                )

                return None

        self.last_error = (
            f"OpenAlex request failed after "
            f"{MAX_RETRIES} attempts."
        )

        return None

    @staticmethod
    def _authors(
        item: Dict[str, Any],
    ) -> List[str]:

        result = []

        for authorship in (
            item.get("authorships", [])
            or []
        ):
            author = authorship.get(
                "author"
            ) or {}

            name = author.get(
                "display_name"
            )

            if name:
                result.append(name)

        return result

    def _to_candidate(
        self,
        item: Dict[str, Any],
        method: str,
        query: str,
    ) -> CandidatePaper:

        primary = (
            item.get("primary_location")
            or {}
        )

        source = (
            primary.get("source")
            or {}
        )

        doi = normalize_doi(
            item.get("doi")
        )

        biblio = (
            item.get("biblio")
            or {}
        )

        first_page = biblio.get(
            "first_page"
        )

        last_page = biblio.get(
            "last_page"
        )

        if first_page and last_page:
            pages = (
                f"{first_page}-{last_page}"
            )
        else:
            pages = first_page

        return CandidatePaper(
            source=self.name,
            source_id=item.get("id"),
            title=item.get("title"),
            authors=self._authors(item),
            year=item.get("publication_year"),
            venue=source.get(
                "display_name"
            ),
            doi=doi,
            volume=biblio.get("volume"),
            issue=biblio.get("issue"),
            pages=pages,
            url=item.get("doi")
            or item.get("id"),
            retrieval_method=method,
            query=query,
            raw=item,
            retrieval_methods=[method],
        )

    def lookup_doi(
        self,
        doi: str,
    ) -> Optional[CandidatePaper]:

        doi = normalize_doi(doi)

        if not doi:
            return None

        payload = self._get(
            "/works/https://doi.org/"
            + doi
        )

        if not payload:
            return None

        return self._to_candidate(
            payload,
            "doi",
            doi,
        )

    def _search(
        self,
        search: str,
        method: str,
        query: str,
    ) -> List[CandidatePaper]:

        if not search:
            return []

        payload = self._get(
            "/works",
            {
                "search": search,
                "per-page": MAX_CANDIDATES_PER_QUERY,
            },
        )

        if not payload:
            return []

        return [
            self._to_candidate(
                item,
                method,
                query,
            )
            for item in payload.get(
                "results",
                [],
            )
        ]

    def search_title(
        self,
        title: str,
    ) -> List[CandidatePaper]:

        return (
            self._search(
                title,
                "title",
                title,
            )
            if title
            else []
        )

    def search_author_title(
        self,
        author: str,
        title: str,
    ) -> List[CandidatePaper]:

        if not title:
            return []

        query = (
            f"{author} {title}"
            .strip()
        )

        return self._search(
            query,
            "author_title",
            query,
        )

    def search_title_venue(
        self,
        title: str,
        venue: str,
    ) -> List[CandidatePaper]:

        if not title:
            return []

        query = (
            f"{title} {venue}"
            .strip()
        )

        return self._search(
            query,
            "title_venue",
            query,
        )