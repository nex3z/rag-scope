from typing import Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.settings.components.embeddings_settings_view import EmbeddingsSettingsView
from rag_scope.ui.settings.components.llm_settings_view import LlmSettingsView
from rag_scope.ui.settings.components.vector_store_settings_view import VectorStoreSettingsView


class SettingsView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None

    def build(self) -> 'SettingsView':
        with gr.Blocks() as self.app:
            with gr.Tab("Embeddings"):
                EmbeddingsSettingsView(app_context=self.app_context).build()
            with gr.Tab("Vector store"):
                VectorStoreSettingsView(app_context=self.app_context).build()
            with gr.Tab("LLM"):
                LlmSettingsView(app_context=self.app_context).build()
        return self


def main():
    app_context = AppContext()
    view = SettingsView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
