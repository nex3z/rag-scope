import json
from typing import Optional

import gradio as gr
from loguru import logger

from rag_scope.retriever.query_vector_store_retriever import QueryVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.retriever.base_retriever_view import BaseRetrieverView
from rag_scope.ui.utils.message_utils import on_load_success
from rag_scope.utils.import_utils import get_qualified_name, import_class

AVAILABLE_RETRIEVERS = [
    get_qualified_name(QueryVectorStoreRetriever),
]
DEFAULT_RETRIEVER = get_qualified_name(QueryVectorStoreRetriever)
DEFAULT_PARAM = {
    'embed_type': 'Query',
    'search_type': 'Similarity',
    'k': 4,
}


class CustomRetrieverView(BaseRetrieverView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self._retriever: Optional[gr.State] = None

        self.retriever_class: Optional[gr.Dropdown] = None
        self.retriever_param: Optional[gr.Code] = None

    @property
    def retriever(self) -> gr.State:
        return self._retriever

    def build(self) -> 'CustomRetrieverView':
        with gr.Blocks() as self.app:
            self._retriever = gr.State(None)

            with gr.Group():
                self.retriever_class = gr.Dropdown(
                    label="Retriever class",
                    choices=AVAILABLE_RETRIEVERS,
                    value=DEFAULT_RETRIEVER,
                    allow_custom_value=True, interactive=True,
                )
                self.retriever_param = gr.Code(
                    label="Retriever param",
                    value=json.dumps(DEFAULT_PARAM, indent=2),
                    lines=5, language='json', interactive=True,
                )

                config_fields = [self.retriever_class, self.retriever_param]

            btn_load = gr.Button("Load", variant='primary')
            btn_load.click(self._on_load_retriever, config_fields, [self._retriever]) \
                .success(on_load_success)

            self.app.load(self._on_load_retriever, config_fields, [self._retriever])

        return self

    def _on_load_retriever(self, retriever_class: str, retriever_param: str):
        if len(retriever_class) == 0:
            raise gr.Error("Missing retriever class.")

        try:
            retriever_param = json.loads(retriever_param)
        except Exception as e:
            raise gr.Error(f"Failed to parse retriever param, error: {e}")
        logger.info(f"retriever_class = {retriever_class}, retriever_param = {retriever_param}")

        try:
            retriever_class = import_class(retriever_class)
            retriever = retriever_class(engine=self.app_context.engine, **retriever_param)
        except Exception as e:
            raise gr.Error(f"Failed to instantiate retriever, error: {e}")
        logger.info(f"Loaded {type(retriever)}: {retriever}")

        return {
            self._retriever: retriever,
        }


def main():
    app_context = AppContext.load_default()
    view = CustomRetrieverView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
