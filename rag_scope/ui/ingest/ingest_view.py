import json
from pathlib import Path
from typing import Optional, List

import gradio as gr
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
# noinspection PyProtectedMember
from langchain_text_splitters import __all__ as all_splitters
from loguru import logger
from tqdm import tqdm

from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.utils.import_utils import instantiate_class, ensure_qualified_name

AVAILABLE_SPLITTERS = all_splitters
DEFAULT_SPLITTER_CLASS = 'RecursiveCharacterTextSplitter'
DEFAULT_SPLITTER_MODULE = 'langchain_text_splitters'
DEFAULT_SPLITTER_PARAM = {
    'chunk_size': 512,
    'chunk_overlap': 20,
}


class IngestView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self.documents: Optional[gr.State] = None

    def build(self) -> 'IngestView':
        with gr.Blocks() as self.app:
            self.documents = gr.State([])

            with gr.Row(equal_height=False):
                with gr.Column():
                    upload_files = gr.File(label='Upload', file_count='multiple')
                with gr.Column(variant='panel'):
                    splitter_class = gr.Dropdown(
                        label='Splitter class',
                        choices=AVAILABLE_SPLITTERS,
                        value=DEFAULT_SPLITTER_CLASS,
                    )
                    splitter_param = gr.Code(
                        label="Splitter param",
                        value=json.dumps(DEFAULT_SPLITTER_PARAM, indent=2),
                        lines=10, language='json', interactive=True
                    )
                    btn_ingest = gr.Button('Ingest', variant='primary')

            gr.on(
                triggers=[upload_files.upload, upload_files.clear],
                fn=self.__on_upload,
                inputs=[upload_files],
                outputs=[self.documents],
            )

            btn_ingest.click(
                self.__on_ingest,
                [splitter_class, splitter_param, self.documents],
                []
            )
        return self

    def __on_upload(self, files: Optional[List[str]]):
        logger.info(f"files = {files}")
        if files is None:
            return {self.documents: []}

        documents = []
        for file in tqdm(files, desc='Loading'):
            file = Path(file)
            if file.suffix == '.txt':
                loader = TextLoader(str(file), encoding='utf-8')
            elif file.suffix == '.pdf':
                loader = PyPDFLoader(str(file))
            else:
                loader = UnstructuredFileLoader(str(file))
            docs = loader.load()
            documents.extend(docs)
        logger.info(f"Loaded {len(documents)} documents.")
        return {self.documents: documents}

    def __on_ingest(self, splitter_class, splitter_param: str, documents: List[Document]):
        if len(documents) == 0:
            gr.Warning("Please upload files first.")
            return

        splitter_param = json.loads(splitter_param)
        splitter_class = ensure_qualified_name(splitter_class, DEFAULT_SPLITTER_MODULE)
        text_splitter: TextSplitter = instantiate_class(splitter_class, splitter_param)

        split_documents = text_splitter.split_documents(documents)
        logger.info(f"Split to {len(split_documents)} chunks.")

        self.app_context.engine.vector_store.add_documents(split_documents)
        logger.info(f"Added to vector store.")

        gr.Info(f"Ingested {len(split_documents)} chunks from {len(documents)} documents.")


def main():
    app_context = AppContext.load_default()
    view = IngestView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
