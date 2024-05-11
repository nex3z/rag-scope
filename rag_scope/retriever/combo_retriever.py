from itertools import chain
from typing import List, Dict
import numpy as np
from loguru import logger
import os

from sentence_transformers import CrossEncoder

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.base import Retrieval, Query
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.retriever.engine import Engine
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings, get_chat_model
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.vector_store.search_type import SearchType

EXPAND_QUERY_PROMPT_TEMPLATE = """You are an AI language model assistant. 
Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {question}"""


EXPAND_ANSWER_PROMPT_TEMPLATE = """You are an AI language model assistant. 
Your task is to provide example answer to the given question. 
By generating answers on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
Question: {question}"""


class ComboRetriever(BaseVectorStoreRetriever):
    expand_query: bool = False
    expand_query_prompt_template: str = EXPAND_QUERY_PROMPT_TEMPLATE

    expand_answer: bool = False
    expand_answer_prompt_template: str = EXPAND_ANSWER_PROMPT_TEMPLATE

    rerank: bool = False
    rerank_cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rerank_top_n: int = 4

    embed_type: EmbedType = EmbedType.QUERY
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 4
    search_kwargs: Dict = dict()

    def retrieve(self, query: str) -> List[Retrieval]:
        if self.expand_query is True:
            logger.info("Generating queries for {query}")
            prompt = self.expand_query_prompt_template.format(question=query)
            response = self.engine.chat_model.invoke(prompt)
            queries = response.content.splitlines()
            logger.info(f"Generated {len(queries)} queries:\n{os.linesep.join(queries)}")
            queries = [query] + queries
        else:
            queries = [query]

        if self.expand_answer is True:
            queries = list(map(self._expand_answer, queries))

        retrievals = [
            self._retrieve(queries[0], 'query'),
        ]

        for generated_query in queries[1:]:
            retrievals.append(self._retrieve(generated_query, 'generated_query'))

        if self.rerank is True:
            documents = list(chain.from_iterable([r.documents for r in retrievals]))
            logger.info(f"Reranking {len(documents)} documents")
            for idx, doc in enumerate(documents):
                doc.metadata['doc_idx'] = idx

            unique_documents, unique_contents = [], set()
            for doc in documents:
                if doc.page_content not in unique_contents:
                    unique_contents.add(doc.page_content)
                    unique_documents.append(doc)
            logger.info(f"Found {len(unique_documents)} unique documents.")

            cross_encoder = CrossEncoder(self.rerank_cross_encoder)
            pairs = [[query, doc.page_content] for doc in unique_documents]
            scores = cross_encoder.predict(pairs)
            idxes = np.argsort(scores)[::-1][:self.rerank_top_n]
            logger.info(f"scores = {scores}, idxes = {idxes}")
            idxes = set(idxes)
            for ret in retrievals:
                ret.documents = list(filter(lambda d: d.metadata['doc_idx'] in idxes, ret.documents))

        return retrievals

    def _expand_answer(self, query: str) -> str:
        prompt = self.expand_answer_prompt_template.format(question=query)
        response = self.engine.chat_model.invoke(prompt)
        joint_query = f"{query} {response.content}"
        logger.info(f"expanded query with answer: {joint_query}")
        return query

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
    retriever = ComboRetriever(engine=engine)

    query = "What is the authors view on the early stages of a startup?"
    documents = retriever.invoke(query)
    print(len(documents))
    print('\n'.join(d.page_content for d in documents))


if __name__ == '__main__':
    main()
