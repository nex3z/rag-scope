from abc import ABC, abstractmethod
from typing import List, Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class EmbeddingExtractor(VectorStore, ABC):
    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        return_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        pass

    @abstractmethod
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        return_embedding: bool = False,
        **kwargs: Any
    ) -> List[Document]:
        pass

    @abstractmethod
    def get_stored_documents(self) -> List[Document]:
        pass
