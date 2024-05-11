from typing import Optional, List

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.embeddings.embeddings_utils import embed_text
from rag_scope.vector_store.search_type import SearchType
from rag_scope.vector_store.vector_store_utils import search_by_vector


class Engine:
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        chat_model: Optional[BaseChatModel] = None,
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.chat_model = chat_model

    def embed_text(self, text: str, embed_type: EmbedType = EmbedType.QUERY):
        return embed_text(self.embeddings, text, embed_type=embed_type)

    def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        search_type: SearchType = SearchType.SIMILARITY,
        **kwargs
    ):
        return search_by_vector(
            self.vector_store,
            embedding,
            k=k,
            search_type=search_type,
            **kwargs,
        )
