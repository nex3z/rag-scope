from typing import Optional

import gradio as gr
from loguru import logger

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.retriever.combo_retriever import ComboRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.retriever.base_retriever_view import BaseRetrieverView
from rag_scope.vector_store.search_type import SearchType


class ComboRetrieverView(BaseRetrieverView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self._retriever: Optional[gr.State] = None

        self.expand_query: Optional[gr.Checkbox] = None
        self.expand_query_prompt_template: Optional[gr.Code] = None

        self.expand_answer: Optional[gr.Checkbox] = None
        self.expand_answer_prompt_template: Optional[gr.Code] = None

        self.rerank: Optional[gr.Checkbox] = None
        self.rerank_cross_encoder: Optional[gr.Dropdown] = None
        self.rerank_top_n: Optional[gr.Dropdown] = None

        self.embed_type: Optional[gr.Radio] = None
        self.search_type: Optional[gr.Radio] = None
        self.k: Optional[gr.Slider] = None

    @property
    def retriever(self) -> gr.State:
        return self._retriever

    def build(self) -> 'ComboRetrieverView':
        with gr.Blocks() as self.app:
            self._retriever = gr.State(lambda: ComboRetriever(engine=self.app_context.engine))

            with gr.Group():
                self.expand_query = gr.Checkbox(
                    label="Expand query",
                    interactive=True)
                self.expand_query_prompt_template = gr.Code(
                    label="Expand query prompt template",
                    lines=5, interactive=True)

            with gr.Group():
                self.expand_answer = gr.Checkbox(
                    label="Expand answer",
                    interactive=True)
                self.expand_answer_prompt_template = gr.Code(
                    label="Expand answer prompt template",
                    lines=5, interactive=True)

            with gr.Group():
                self.rerank = gr.Checkbox(
                    label="Rerank",
                    interactive=True)
                with gr.Row():
                    self.rerank_cross_encoder = gr.Dropdown(
                        label="Cross encoder",
                        choices=['cross-encoder/ms-marco-MiniLM-L-6-v2'],
                        allow_custom_value=True, interactive=True)
                    self.rerank_top_n = gr.Slider(
                        label="Top N",
                        minimum=1, maximum=100, interactive=True)

            with gr.Group():
                with gr.Row():
                    # noinspection PyUnresolvedReferences
                    self.embed_type = gr.Radio(
                        label="Embed type",
                        choices=[e.value for e in EmbedType],
                        interactive=True)

                    # noinspection PyUnresolvedReferences
                    self.search_type = gr.Radio(
                        label="Search type",
                        choices=[e.value for e in SearchType],
                        interactive=True)

                self.k = gr.Slider(
                    label="k",
                    minimum=1, maximum=100, step=1,
                    interactive=True)

            config_fields = [
                self.expand_query, self.expand_query_prompt_template,
                self.expand_answer, self.expand_answer_prompt_template,
                self.rerank, self.rerank_cross_encoder, self.rerank_top_n,
                self.embed_type, self.search_type, self.k,
            ]

            gr.on(
                [f.change for f in config_fields],
                self._on_change,
                [self._retriever] + config_fields,
                [self._retriever] + config_fields,
                show_progress='hidden',
            )

            self.app.load(self._on_load, [self._retriever], config_fields)

        return self

    def _on_load(self, retriever: ComboRetriever):
        return {
            self.expand_query: retriever.expand_query,
            self.expand_query_prompt_template: retriever.expand_query_prompt_template,

            self.expand_answer: retriever.expand_answer,
            self.expand_answer_prompt_template: retriever.expand_answer_prompt_template,

            self.rerank: retriever.rerank,
            self.rerank_cross_encoder: retriever.rerank_cross_encoder,
            self.rerank_top_n: retriever.rerank_top_n,

            self.embed_type: retriever.embed_type.value,
            self.search_type: retriever.search_type.value,
            self.k: retriever.k,
        }

    def _on_change(
        self,
        retriever: ComboRetriever,
        expand_query: bool, expand_query_prompt_template: str,
        expand_answer: bool, expand_answer_prompt_template: str,
        rerank: bool, rerank_cross_encoder: str, rerank_top_n: int,
        embed_type: str, search_type: str, k: int,
    ):
        logger.info(
            f"expand_query = {expand_query}, expand_query_prompt_template = {expand_query_prompt_template}, "
            f"expand_answer = {expand_answer}, expand_answer_prompt_template = {expand_answer_prompt_template}, "
            f"rerank = {rerank}, rerank_cross_encoder = {rerank_cross_encoder}, rerank_top_n = {rerank_top_n}, "
            f"embed_type = {embed_type}, search_type = {search_type}, k = {k}"
        )

        retriever.expand_query = expand_query
        retriever.expand_query_prompt_template = expand_query_prompt_template
        retriever.expand_answer = expand_answer
        retriever.expand_answer_prompt_template = expand_answer_prompt_template
        retriever.rerank = rerank
        retriever.rerank_cross_encoder = rerank_cross_encoder
        retriever.rerank_top_n = rerank_top_n
        retriever.embed_type = EmbedType(embed_type)
        retriever.search_type = SearchType(search_type)
        retriever.k = k

        return {
            self._retriever: retriever,
            self.expand_query_prompt_template: gr.Code(interactive=expand_query),
            self.expand_answer_prompt_template: gr.Code(interactive=expand_answer),
            self.rerank_cross_encoder: gr.Code(interactive=rerank),
            self.rerank_top_n: gr.Code(interactive=rerank),
        }


def main():
    app_context = AppContext.load_default()
    view = ComboRetrieverView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
