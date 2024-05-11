from itertools import chain
from typing import List

import pandas as pd
from langchain_core.documents import Document

from rag_scope.retriever.base import Retrieval


def format_retrievals_to_data_frame(retrievals: List[Retrieval]) -> pd.DataFrame:
    documents = list(chain.from_iterable(r.documents for r in retrievals))
    return format_documents_to_data_frame(documents)


def format_documents_to_data_frame(documents: List[Document]) -> pd.DataFrame:
    records = []
    for document in documents:
        record = dict()
        metadata = document.metadata
        if 'score' in metadata:
            record['score'] = metadata['score']
        record['page_content'] = document.page_content
        records.append(record)
    df = pd.DataFrame(records)
    return df


def format_documents_to_string(docs: List[Document]) -> str:
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted


def get_embeddings_from_documents(documents: List[Document]) -> List[List[float]]:
    return [d.metadata['embedding'] for d in documents]


def get_texts_from_documents(documents: List[Document]) -> List[str]:
    return [d.page_content for d in documents]
