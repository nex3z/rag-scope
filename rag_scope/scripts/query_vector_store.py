from langchain_community.vectorstores.chroma import Chroma
from loguru import logger

from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.model_utils import get_embeddings


def main():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    query = "What is the authors view on the early stages of a startup?"
    # docs = db.similarity_search(query)
    # docs = db.similarity_search_with_score(query)
    embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector_with_relevance_scores(embedding)
    logger.info(f"docs = {docs}")

    docs = db.similarity_search_with_score(query)
    logger.info(f"docs = {docs}")

    docs = db.similarity_search_with_relevance_scores(query)
    logger.info(f"docs = {docs}")


if __name__ == '__main__':
    main()
