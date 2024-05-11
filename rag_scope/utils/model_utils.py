import torch
from langchain.schema.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import SecretStr

EMBEDDING_BGE_SMALL_ZH = 'BAAI/bge-small-zh-v1.5'
EMBEDDING_BGE_LARGE_ZH = 'BAAI/bge-large-zh-v1.5'
EMBEDDING_BGE_SMALL_EN = 'BAAI/bge-small-en-v1.5'
EMBEDDING_BGE_LARGE_EN = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DEFAULT = EMBEDDING_BGE_SMALL_EN


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_chat_model() -> BaseChatModel:
    return ChatOpenAI(api_key='dummy', base_url='http://127.0.0.1:8000/v1')


def get_embeddings(
    model_name: str = EMBEDDING_DEFAULT,
    device: str = get_device()
) -> Embeddings:
    logger.info(f"Loading embeddings {model_name}, device = {device}")
    # noinspection Pydantic,PyArgumentList
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Loaded embeddings")
    return embeddings
