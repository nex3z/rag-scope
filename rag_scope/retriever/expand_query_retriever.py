import os
from typing import Dict, List

from loguru import logger

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.base import Retrieval, Query
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.retriever.engine import Engine
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings, get_chat_model
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.vector_store.search_type import SearchType

QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}"""


class ExpandQueryRetriever(BaseVectorStoreRetriever):
    prompt_template: str = QUERY_PROMPT

    embed_type: EmbedType = EmbedType.QUERY
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 4

    search_kwargs: Dict = dict()

    def retrieve(self, query: str) -> List[Retrieval]:
        prompt = self.prompt_template.format(question=query)
        response = self.engine.chat_model.invoke(prompt)
        generated_queries = response.content.splitlines()
        logger.info(f"Generated {len(generated_queries)} queries:\n{os.linesep.join(generated_queries)}")

        retrievals = [
            self._retrieve(query, 'query'),
        ]

        for generated_query in generated_queries:
            retrievals.append(self._retrieve(generated_query, 'generated_query'))

        return retrievals

    def _retrieve(self, query: str, query_type: str) -> Retrieval:
        query_embedding = self.engine.embed_text(query, embed_type=self.embed_type)
        documents = self.engine.search_by_vector(
            query_embedding,
            k=self.k,
            search_type=self.search_type,
            **self.search_kwargs,
        )
        return Retrieval(
            query=Query.from_query(query, query_embedding, query_type),
            documents=documents,
        )


def main():
    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(persist_directory=DB_DIR, embedding_function=embeddings, return_embedding=True)
    chat_model = get_chat_model()
    engine = Engine(
        embeddings=embeddings,
        vector_store=vector_store,
        chat_model=chat_model,
    )
    retriever = ExpandQueryRetriever(engine=engine)

    query = "What is the authors view on the early stages of a startup?"
    documents = retriever.invoke(query)
    print(len(documents))
    print(documents)


if __name__ == '__main__':
    main()
