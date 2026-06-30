from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.normalized_candidate import NormalizedCandidate


class BaseNormalizer(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def normalize(self, candidate: NormalizedCandidate) -> None:
        ...