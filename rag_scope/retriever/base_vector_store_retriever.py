from abc import ABC, abstractmethod
from itertools import chain
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_scope.retriever.base import Retrieval
from rag_scope.retriever.engine import Engine


class BaseVectorStoreRetriever(BaseRetriever, ABC):
    engine: Engine

    @abstractmethod
    def retrieve(self, query: str) -> List[Retrieval]:
        pass

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        retrievals = self.retrieve(query)
        return list(chain.from_iterable(r.documents for r in retrievals))
