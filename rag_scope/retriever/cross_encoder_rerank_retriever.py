from typing import Dict, List, Optional

from loguru import logger
from sentence_transformers import CrossEncoder
import numpy as np
from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.base import Retrieval, Query
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.retriever.engine import Engine
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.vector_store.search_type import SearchType


class CrossEncoderRerankRetriever(BaseVectorStoreRetriever):
    embed_type: EmbedType = EmbedType.QUERY
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 4

    cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    limit: Optional[int] = 3

    search_kwargs: Dict = dict()

    def retrieve(self, query: str) -> List[Retrieval]:
        query_embedding = self.engine.embed_text(query, embed_type=self.embed_type)
        documents = self.engine.search_by_vector(
            query_embedding,
            k=self.k,
            search_type=self.search_type,
            **self.search_kwargs,
        )

        if self.limit is not None:
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[query, doc.page_content] for doc in documents]
            scores = cross_encoder.predict(pairs)
            idxes = np.argsort(scores)[::-1][:self.limit]
            logger.info(f"scores = {scores}, idxes = {idxes}")
            documents = [documents[i] for i in idxes]

        return [Retrieval(
            query=Query.from_query(query, query_embedding),
            documents=documents,
        )]


def main():
    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(persist_directory=DB_DIR, embedding_function=embeddings, return_embedding=True)
    engine = Engine(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    retriever = CrossEncoderRerankRetriever(engine=engine)
    query = "What is the authors view on the early stages of a startup?"
    documents = retriever.invoke(query)
    print(documents)


if __name__ == '__main__':
    main()
