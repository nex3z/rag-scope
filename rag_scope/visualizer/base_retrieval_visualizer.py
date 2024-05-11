from abc import ABC, abstractmethod
from typing import List, Optional

from rag_scope.retriever.base import Retrieval


class BaseRetrievalVisualizer(ABC):

    @abstractmethod
    def render(
        self,
        retrievals: List[Retrieval],
        limit: Optional[int] = None,
    ):
        pass
