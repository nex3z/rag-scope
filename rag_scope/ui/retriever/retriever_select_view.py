from typing import Optional

import gradio as gr
from loguru import logger

from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.retriever.base_retriever_view import BaseRetrieverView
from rag_scope.ui.retriever.combo_retriever_view import ComboRetrieverView
from rag_scope.ui.retriever.custom_retriever_view import CustomRetrieverView

TAB_BUILTIN = "Builtin"
TAB_CUSTOM = "Custom"


class RetrieverSelectView(BaseRetrieverView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self._retriever: Optional[gr.State] = None
        self.tab_builtin: Optional[gr.Tab] = None
        self.tab_custom: Optional[gr.Tab] = None

    @property
    def retriever(self) -> gr.State:
        return self._retriever

    def build(self) -> 'RetrieverSelectView':
        with gr.Blocks() as self.app:
            self._retriever = gr.State(None)

            with gr.Tab(TAB_BUILTIN) as self.tab_builtin:
                builtin_retriever_view = ComboRetrieverView(app_context=self.app_context)
                builtin_retriever_view.build()
            with gr.Tab(TAB_CUSTOM) as self.tab_custom:
                custom_retriever_view = CustomRetrieverView(app_context=self.app_context)
                custom_retriever_view.build()

            tabs = [self.tab_builtin, self.tab_custom]
            gr.on(
                [tab.select for tab in tabs],
                self.__on_tab_selected,
                [builtin_retriever_view.retriever, custom_retriever_view.retriever],
                [self._retriever],
            )

            self.app.load(
                self.__init_retriever_provider,
                [builtin_retriever_view.retriever],
                [self._retriever]
            )

        return self

    def __init_retriever_provider(self, default_retriever: BaseVectorStoreRetriever):
        logger.info(f"Initializing retriever = {default_retriever}")
        return {
            self._retriever: default_retriever,
        }

    def __on_tab_selected(
        self,
        builtin_retriever: BaseVectorStoreRetriever, custom_retriever: BaseVectorStoreRetriever,
        evt: gr.SelectData
    ):
        logger.info(f"evt.value = {evt.value}")
        if evt.value == TAB_BUILTIN:
            provider = builtin_retriever
        elif evt.value == TAB_CUSTOM:
            provider = custom_retriever
        else:
            raise ValueError(f"Invalid tab {evt.value}")
        logger.info(f"Selected provider = {provider}")
        return {
            self._retriever: provider
        }


def main():
    app_context = AppContext.load_default()
    view = RetrieverSelectView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
