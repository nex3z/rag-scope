import uuid
from pathlib import Path
from typing import List, Type, Optional, Any, Iterable, Dict, Tuple, Union

import chromadb
import numpy as np
from chromadb import QueryResult
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST
from loguru import logger

from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings
from rag_scope.vector_store.embedding_extractor import EmbeddingExtractor

_LANGCHAIN_DEFAULT_COLLECTION_NAME = 'langchain'
DEFAULT_K = 4
INCLUDE_FIELDS = ['documents', 'metadatas', 'distances']
INCLUDE_EMBEDDINGS_FIELDS = ['documents', 'metadatas', 'distances', 'embeddings']


class ChromaVectorStore(EmbeddingExtractor):
    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[Union[Path, str]] = None,
        # client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
        # client: Optional[chromadb.Client] = None,
        # relevance_score_fn: Optional[Callable[[float], float]] = None,
        return_embedding: bool = False,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
        )
        self._return_embedding = return_embedding

    # noinspection PyShadowingBuiltins
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=where_document,
            return_embedding=self._return_embedding,
            **kwargs,
        )
        return _results_to_docs(results)

    # noinspection PyShadowingBuiltins
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,

        **kwargs: Any,
    ) -> List[Document]:
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            where_document=where_document,
            return_embedding=True,
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs(results, include_embeddings=self._return_embedding)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    # noinspection PyShadowingBuiltins
    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, return_embedding=self._return_embedding, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    # noinspection PyShadowingBuiltins
    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
                return_embedding=self._return_embedding,
                **kwargs,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
                return_embedding=self._return_embedding,
                **kwargs,
            )

        return _results_to_docs_and_scores(results)

    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        n_results: int = DEFAULT_K,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        return_embedding: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        include = INCLUDE_EMBEDDINGS_FIELDS if return_embedding is True else INCLUDE_FIELDS
        # noinspection PyTypeChecker
        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            **kwargs,
        )

    def get_stored_documents(self) -> List[Document]:
        result = self._collection.get(include=['embeddings', 'documents', 'metadatas'])
        documents = []
        for i in range(len(result['ids'])):
            metadata = dict()
            if result['metadatas'][i] is not None:
                metadata.update(result['metadatas'][i])
            metadata['id'] = result['ids'][i]

            if ('embeddings' in result) and (result['embeddings'] is not None):
                metadata['embedding'] = result['embeddings'][i]
            else:
                logger.warning("Missing embeddings in result.")

            document = Document(
                page_content=result['documents'][i],
                metadata=metadata,
            )
            documents.append(document)
        return documents

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = (
                            "Try filtering complex metadata from the document using "
                            "langchain_community.vectorstores.utils.filter_complex_metadata."
                        )
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

    @classmethod
    def from_texts(cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None,
                   **kwargs: Any) -> VST:
        raise NotImplementedError


def _results_to_docs(results: Any, include_embeddings: bool = True) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results, include_embeddings=include_embeddings)]


def _results_to_docs_and_scores(result: QueryResult, include_embeddings: bool = True) -> List[Tuple[Document, float]]:
    documents = []
    for i in range(len(result['ids'][0])):
        metadata = dict()
        if result['metadatas'][0][i] is not None:
            metadata.update(result['metadatas'][0][i])
        metadata['score'] = result['distances'][0][i]
        metadata['id'] = result['ids'][0][i]

        if (include_embeddings is True) and ('embeddings' in result) and (result['embeddings'] is not None):
            metadata['embedding'] = result['embeddings'][0][i]
        else:
            logger.warning("Missing embeddings in result.")

        document = Document(
            page_content=result['documents'][0][i],
            metadata=metadata,
        )
        documents.append((document, result['distances'][0][i]))
    return documents


def main():
    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(persist_directory=DB_DIR, embedding_function=embeddings)

    query = "What is the authors view on the early stages of a startup?"
    embedding = embeddings.embed_query(query)

    docs = vector_store.similarity_search(query, return_embedding=True)
    logger.info(f"docs = {docs}")

    docs = vector_store.similarity_search_by_vector(embedding, return_embedding=True)
    logger.info(f"docs = {docs}")

    docs = vector_store.max_marginal_relevance_search_by_vector(embedding, return_embedding=True)
    logger.info(f"docs = {docs}")


if __name__ == '__main__':
    main()
