from typing import List, Dict

from loguru import logger

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.base import Retrieval, Query
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.retriever.engine import Engine
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings, get_chat_model
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.vector_store.search_type import SearchType

PROMPT_TEMPLATE = """You are an AI language model assistant. Your task is to provide example answer to the given
question. By generating answers on the user question, your goal is to help the user overcome some of the limitations 
of the distance-based similarity search. 
Question: {question}"""


class ExpandAnswerRetriever(BaseVectorStoreRetriever):
    prompt_template: str = PROMPT_TEMPLATE

    embed_type: EmbedType = EmbedType.QUERY
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 4
    search_kwargs: Dict = dict()

    def retrieve(self, query: str) -> List[Retrieval]:
        prompt = self.prompt_template.format(question=query)
        response = self.engine.chat_model.invoke(prompt)
        joint_query = f"{query} {response.content}"
        logger.info(f"joint_query = {joint_query}")

        joint_query_embedding = self.engine.embed_text(joint_query, embed_type=self.embed_type)
        documents = self.engine.search_by_vector(
            joint_query_embedding,
            k=self.k,
            search_type=self.search_type,
            **self.search_kwargs,
        )

        query_embedding = self.engine.embed_text(query, embed_type=self.embed_type)
        return [
            Retrieval(
                query=Query.from_query(query, query_embedding),
            ),
            Retrieval(
                query=Query.from_query(joint_query, joint_query_embedding, 'query_with_answer'),
                documents=documents,
            ),
        ]


def main():
    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(persist_directory=DB_DIR, embedding_function=embeddings, return_embedding=True)
    chat_model = get_chat_model()
    engine = Engine(
        embeddings=embeddings,
        vector_store=vector_store,
        chat_model=chat_model,
    )
    retriever = ExpandAnswerRetriever(engine=engine)

    query = "What is the authors view on the early stages of a startup?"
    documents = retriever.invoke(query)
    print(len(documents))
    print(documents)


if __name__ == '__main__':
    main()
