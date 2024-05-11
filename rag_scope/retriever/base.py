from typing import List, Dict

from langchain_core.documents import Document
from pydantic import v1


class Query(v1.BaseModel):
    query_content: str
    metadata: Dict = dict()
    type: str = 'query'

    # noinspection PyShadowingBuiltins
    @staticmethod
    def from_query(query_content: str, embedding: List[float], type: str = 'query'):
        return Query(
            query_content=query_content,
            type=type,
            metadata={
                'embedding': embedding,
            }
        )


class Retrieval(v1.BaseModel):
    query: Query
    documents: List[Document] = []
