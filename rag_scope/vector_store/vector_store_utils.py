from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rag_scope.vector_store.search_type import SearchType


def search_by_vector(
    vector_store: VectorStore,
    embedding: List[float],
    k: int = 4,
    search_type: SearchType = SearchType.SIMILARITY,
    **kwargs
) -> List[Document]:
    if search_type == SearchType.SIMILARITY:
        return vector_store.similarity_search_by_vector(embedding, k=k, **kwargs)
    elif search_type == SearchType.MMR:
        return vector_store.max_marginal_relevance_search_by_vector(embedding, k=k, **kwargs)
    else:
        raise ValueError(f"Unknown search type: {search_type}")
