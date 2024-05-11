from typing import List

from langchain_core.embeddings import Embeddings

from rag_scope.embeddings.embed_type import EmbedType


def embed_text(embeddings: Embeddings, text: str, embed_type: EmbedType = EmbedType.QUERY) -> List[float]:
    if embed_type == EmbedType.QUERY:
        return embeddings.embed_query(text)
    elif embed_type == EmbedType.DOCUMENT:
        return embeddings.embed_documents([text])[0]
    else:
        raise ValueError(f"Unknown embedding method {embed_type}")
