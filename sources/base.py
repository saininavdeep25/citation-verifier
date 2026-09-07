from abc import ABC, abstractmethod
from typing import List, Optional

from models.schemas import CandidatePaper, CitationMetadata


class ScholarlySource(ABC):
    name = "base"

    @abstractmethod
    def lookup_doi(self, doi: str) -> Optional[CandidatePaper]:
        raise NotImplementedError

    @abstractmethod
    def search_title(self, title: str) -> List[CandidatePaper]:
        raise NotImplementedError

    @abstractmethod
    def search_author_title(self, author: str, title: str) -> List[CandidatePaper]:
        raise NotImplementedError

    @abstractmethod
    def search_title_venue(self, title: str, venue: str) -> List[CandidatePaper]:
        raise NotImplementedError
