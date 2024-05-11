from typing import Optional, Dict, List, Iterator, Sequence

from langchain_community.chat_models import __all__ as all_chat_models
from langchain_community.embeddings import __all__ as all_embeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import ConfigDict, BaseModel, Field

from rag_scope.retriever.engine import Engine
from rag_scope.scripts.build_vector_store import DB_DIR
from rag_scope.utils.import_utils import instantiate_class, import_class, get_qualified_name, ensure_qualified_name
from rag_scope.vector_store.chroma.chroma_vector_store import ChromaVectorStore
from rag_scope.vector_store.embedding_extractor import EmbeddingExtractor

AVAILABLE_EMBEDDINGS_CLASSES = all_embeddings
DEFAULT_EMBEDDINGS_MODULE = 'langchain_community.embeddings'
DEFAULT_EMBEDDINGS_CLASS = 'HuggingFaceBgeEmbeddings'
DEFAULT_EMBEDDINGS_PARAM = {
    'model_name': 'BAAI/bge-small-en-v1.5',
    'model_kwargs': {'device': 'cpu'},
    'encode_kwargs': {'normalize_embeddings': True}
}

AVAILABLE_VECTOR_STORE_CLASSES = [get_qualified_name(ChromaVectorStore)]
DEFAULT_VECTOR_STORE_MODULE = 'langchain_community.vectorstores'
DEFAULT_VECTOR_STORE_CLASS = get_qualified_name(ChromaVectorStore)
DEFAULT_VECTOR_STORE_PARAM = {
    'persist_directory': str(DB_DIR),
    'return_embedding': True,
}

AVAILABLE_CHAT_MODELS = [get_qualified_name(ChatOpenAI)] + all_chat_models
DEFAULT_CHAT_MODEL_MODULE = 'langchain_community.chat_models'
DEFAULT_CHAT_MODEL = get_qualified_name(ChatOpenAI)
DEFAULT_CHAT_MODEL_PARAM = {
    'openai_api_key': 'dummy',
    'openai_api_base': 'http://127.0.0.1:8000/v1'
}


class AppContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings_class: str = DEFAULT_EMBEDDINGS_CLASS
    embeddings_param: Dict = DEFAULT_EMBEDDINGS_PARAM

    vector_store_class: str = DEFAULT_VECTOR_STORE_CLASS
    vector_store_param: Dict = DEFAULT_VECTOR_STORE_PARAM

    chat_model_class: str = DEFAULT_CHAT_MODEL
    chat_model_param: Dict = DEFAULT_CHAT_MODEL_PARAM

    engine: Engine = Field(Engine(), exclude=True)

    def load_embeddings(
        self,
        embeddings_class: Optional[str] = None,
        embeddings_param: Optional[Dict] = None
    ):
        embeddings_class = embeddings_class if embeddings_class is not None else self.embeddings_class
        embeddings_class = ensure_qualified_name(embeddings_class, DEFAULT_EMBEDDINGS_MODULE)
        embeddings_param = embeddings_param if embeddings_param is not None else self.embeddings_param
        logger.info(f"Loading embeddings_class = {embeddings_class}, embeddings_param = {embeddings_param}")

        self.engine.embeddings = instantiate_class(embeddings_class, embeddings_param)
        self.embeddings_class = embeddings_class
        self.embeddings_param = embeddings_param
        logger.info("Loaded embeddings")

    def load_vector_store(
        self,
        vector_store_class: Optional[str] = None,
        vector_store_param: Optional[Dict] = None,
    ):
        vector_store_class = vector_store_class if vector_store_class is not None else self.vector_store_class
        vector_store_class = ensure_qualified_name(vector_store_class, DEFAULT_VECTOR_STORE_MODULE)
        vector_store_param = vector_store_param if vector_store_param is not None else self.vector_store_param
        logger.info(f"Loading vector_store_class = {vector_store_class}, vector_store_param = {vector_store_param}")

        vector_store = import_class(vector_store_class)
        self.engine.vector_store = vector_store(
            embedding_function=self.engine.embeddings,
            **vector_store_param
        )
        self.vector_store_class = vector_store_class
        self.vector_store_param = vector_store_param
        logger.info("Loaded vector store.")

    def get_stored_documents(self) -> List[Document]:
        if isinstance(self.engine.vector_store, EmbeddingExtractor):
            return self.engine.vector_store.get_stored_documents()
        else:
            logger.warning(f"get_stored_documents is not supported on {self.engine.vector_store}")
            return []

    def load_chat_model(
        self,
        chat_model_class: Optional[str] = None,
        chat_model_param: Optional[Dict] = None,
    ):
        chat_model_class = chat_model_class if chat_model_class is not None else self.chat_model_class
        chat_model_class = ensure_qualified_name(chat_model_class, DEFAULT_CHAT_MODEL_MODULE)
        chat_model_param = chat_model_param if chat_model_param is not None else self.chat_model_param
        logger.info(f"Loading chat_model_class = {chat_model_class}")

        self.engine.chat_model = instantiate_class(chat_model_class, chat_model_param)
        self.chat_model_class = chat_model_class
        self.chat_model_param = chat_model_param
        logger.info("Loaded chat model.")

    def chat(self, user_input: str) -> str:
        message = self.engine.chat_model.invoke(user_input)
        return message.content

    def stream_chat(self, user_input: str) -> Iterator[str]:
        response = self.engine.chat_model.stream(user_input)
        for chunk in response:
            yield chunk.content

    def stream(self, messages: Sequence[BaseMessage]):
        response = self.engine.chat_model.stream(messages)
        for chunk in response:
            yield chunk.content

    @staticmethod
    def load_default() -> 'AppContext':
        app_context = AppContext()
        app_context.load_embeddings()
        app_context.load_vector_store()
        app_context.load_chat_model()
        return app_context


def main():
    app_context = AppContext.load_default()

    response = app_context.stream_chat("Hi")
    for chunk in response:
        print(chunk)


if __name__ == '__main__':
    main()
