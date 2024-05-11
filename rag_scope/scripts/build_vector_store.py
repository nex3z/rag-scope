import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from loguru import logger

from rag_scope import DATA_ROOT
from rag_scope.utils.model_utils import get_embeddings

DB_DIR = DATA_ROOT / 'chroma' / 'db_sample'


def main():
    build_vector_store(reset=True)


def build_vector_store(reset: bool = False):
    if reset is True and DB_DIR.exists() is True:
        shutil.rmtree(DB_DIR)

    text_file = DATA_ROOT / 'documents' / 'paul_graham_essay.txt'

    logger.info(f"Loading {text_file}")
    loader = TextLoader(str(text_file), encoding='utf-8')
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"Split to {len(split_documents)} documents.")

    embeddings = get_embeddings()
    Chroma.from_documents(split_documents, embeddings, persist_directory=str(DB_DIR))


if __name__ == '__main__':
    main()
