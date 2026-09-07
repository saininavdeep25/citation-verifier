import logging
import random
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from config import (
    BACKOFF_BASE_SECONDS,
    CROSSREF_BASE_URL,
    CROSSREF_EMAIL,
    MAX_BACKOFF_SECONDS,
    MAX_CANDIDATES_PER_QUERY,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    USER_AGENT,
)
from models.schemas import CandidatePaper
from sources.base import ScholarlySource
from utils.normalization import normalize_doi


logger = logging.getLogger("citation_verifier")


class CrossrefSource(ScholarlySource):
    name = "Crossref"

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

    def _params(
        self,
        rows: int = MAX_CANDIDATES_PER_QUERY,
    ) -> Dict[str, Any]:

        params = {
            "rows": rows
        }

        if CROSSREF_EMAIL:
            params["mailto"] = CROSSREF_EMAIL

        return params

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:

        self._reset_error()

        for attempt in range(MAX_RETRIES):

            try:
                response = self.session.get(
                    f"{CROSSREF_BASE_URL}{path}",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                self.last_status_code = (
                    response.status_code
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (
                    self.RETRYABLE_STATUS_CODES
                ):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:
                        try:
                            delay = float(
                                retry_after
                            )
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
                        "Crossref returned HTTP %s. "
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
                    "Crossref request failed: %s",
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
                    "Invalid JSON returned by Crossref: "
                    + str(exc)
                )

                logger.error(
                    "Crossref returned invalid JSON: %s",
                    exc,
                )

                return None

        self.last_error = (
            f"Crossref request failed after "
            f"{MAX_RETRIES} attempts."
        )

        return None

    @staticmethod
    def _authors(
        item: Dict[str, Any],
    ) -> List[str]:

        result = []

        for author in (
            item.get("author", [])
            or []
        ):

            family = author.get(
                "family"
            )

            given = author.get(
                "given"
            )

            if family and given:
                result.append(
                    f"{given} {family}"
                )

            elif family:
                result.append(family)

            elif given:
                result.append(given)

        return result

    def _to_candidate(
        self,
        item: Dict[str, Any],
        method: str,
        query: str,
    ) -> CandidatePaper:

        date_parts = (
            item.get(
                "published-print",
                {},
            ).get("date-parts")
            or item.get(
                "published-online",
                {},
            ).get("date-parts")
            or item.get(
                "issued",
                {},
            ).get("date-parts")
            or []
        )

        year = (
            date_parts[0][0]
            if date_parts
            and date_parts[0]
            else None
        )

        title_list = (
            item.get("title")
            or []
        )

        venue_list = (
            item.get(
                "container-title"
            )
            or []
        )

        doi = normalize_doi(
            item.get("DOI")
        )

        url = (
            item.get("URL")
            or (
                f"https://doi.org/{doi}"
                if doi
                else None
            )
        )

        return CandidatePaper(
            source=self.name,
            source_id=doi,
            title=(
                title_list[0]
                if title_list
                else None
            ),
            authors=self._authors(item),
            year=year,
            venue=(
                venue_list[0]
                if venue_list
                else None
            ),
            doi=doi,
            volume=item.get("volume"),
            issue=item.get("issue"),
            pages=item.get("page"),
            url=url,
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
            f"/works/{quote(doi, safe='')}"
        )

        if not payload:
            return None

        item = payload.get(
            "message"
        )

        if not item:
            return None

        return self._to_candidate(
            item,
            "doi",
            doi,
        )

    def _search(
        self,
        params: Dict[str, Any],
        method: str,
        query: str,
    ) -> List[CandidatePaper]:

        payload = self._get(
            "/works",
            params=params,
        )

        if not payload:
            return []

        items = (
            payload
            .get("message", {})
            .get("items", [])
        )

        return [
            self._to_candidate(
                item,
                method,
                query,
            )
            for item in items
        ]

    def search_title(
        self,
        title: str,
    ) -> List[CandidatePaper]:

        if not title:
            return []

        params = self._params()

        params["query.title"] = title

        return self._search(
            params,
            "title",
            title,
        )

    def search_author_title(
        self,
        author: str,
        title: str,
    ) -> List[CandidatePaper]:

        if not title:
            return []

        params = self._params()

        params["query.title"] = title

        if author:
            params["query.author"] = author

        query = (
            f"{author} + {title}"
            if author
            else title
        )

        return self._search(
            params,
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

        params = self._params()

        params["query.title"] = title

        if venue:
            params["query.container-title"] = venue

        query = (
            f"{title} + {venue}"
            if venue
            else title
        )

        return self._search(
            params,
            "title_venue",
            query,
        )