from typing import Optional

import gradio as gr
from loguru import logger

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.query_vector_store_retriever import QueryVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.retriever.base_retriever_view import BaseRetrieverView
from rag_scope.vector_store.search_type import SearchType


class VectorStoreRetrieverView(BaseRetrieverView):
    def __init__(self, app_context: AppContext, ):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self._retriever: Optional[gr.State] = None

        self.embed_type: Optional[gr.Radio] = None
        self.search_type: Optional[gr.Radio] = None
        self.k: Optional[gr.Slider] = None

    @property
    def retriever(self) -> gr.State:
        return self._retriever

    def build(self) -> 'VectorStoreRetrieverView':
        with gr.Blocks() as self.app:
            self._retriever = gr.State(lambda: QueryVectorStoreRetriever(engine=self.app_context.engine))

            with gr.Group():
                with gr.Row():
                    # noinspection PyUnresolvedReferences
                    self.embed_type = gr.Radio(
                        label="Embed type",
                        choices=[e.value for e in EmbedType],
                        interactive=True,
                    )

                    # noinspection PyUnresolvedReferences
                    self.search_type = gr.Radio(
                        label="Search type",
                        choices=[e.value for e in SearchType],
                        interactive=True,
                    )

                self.k = gr.Slider(
                    label="k",
                    minimum=1, maximum=100, step=1,
                    interactive=True,
                )

            config_fields = [self.embed_type, self.search_type, self.k]

            gr.on(
                [f.change for f in config_fields],
                self._on_change,
                [self.retriever] + config_fields,
                [self.retriever],
                show_progress='hidden',
            )

            self.app.load(self._on_load, [self._retriever], config_fields)

        return self

    def _on_load(self, retriever: QueryVectorStoreRetriever):
        return {
            self.embed_type: retriever.embed_type.value,
            self.search_type: retriever.search_type.value,
            self.k: retriever.k,
        }

    def _on_change(
        self,
        retriever: QueryVectorStoreRetriever,
        embed_type: str, search_type: str, k: int,
    ):
        logger.info(f"embed_type = {embed_type}, search_type = {search_type}, k = {k}")
        retriever.embed_type = EmbedType(embed_type)
        retriever.search_type = SearchType(search_type)
        retriever.k = k
        return {
            self._retriever: retriever,
        }


def main():
    app_context = AppContext()
    view = VectorStoreRetrieverView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
