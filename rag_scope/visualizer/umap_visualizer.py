from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from numpy.typing import NDArray
from umap import UMAP

from rag_scope.retriever.engine import Engine
from rag_scope.retriever.query_vector_store_retriever import QueryVectorStoreRetriever
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings, get_chat_model
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.visualizer.model_based_visualizer import ModelBasedVisualizer


class UmapVisualizer(ModelBasedVisualizer):
    def __init__(self, stored: List[Document], n_components: int = 2, random_state: Optional[int] = None):
        super().__init__(stored=stored, random_state=random_state)
        self.n_components = n_components
        self.model: Optional[UMAP] = None

    def transform(self, embeddings: NDArray) -> NDArray:
        if self.model is None:
            self.model = UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                transform_seed=0,
            )
            self.model.fit(self.stored_embeddings)

        return self.model.transform(embeddings)


def main():
    pd.options.display.width = 0

    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(persist_directory=DB_DIR, embedding_function=embeddings, return_embedding=True)
    chat_model = get_chat_model()
    engine = Engine(
        embeddings=embeddings,
        vector_store=vector_store,
        chat_model=chat_model,
    )
    retriever = QueryVectorStoreRetriever(engine=engine)
    query = "What is the authors view on the early stages of a startup?"
    retrievals = retriever.retrieve(query)
    # print(retrievals)

    viz = UmapVisualizer(vector_store.get_stored_documents(), n_components=2, random_state=42)
    fig = viz.render(retrievals)
    fig.show()


if __name__ == '__main__':
    main()
