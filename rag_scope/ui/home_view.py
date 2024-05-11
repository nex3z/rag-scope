from typing import Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.chatbot.chatbot_view import ChatbotView
from rag_scope.ui.ingest.ingest_view import IngestView
from rag_scope.ui.settings.settings_view import SettingsView
from rag_scope.ui.vector_store.vector_store_view import VectorStoreView

CSS = """
gradio-app > .gradio-container {
    max-width: 100% !important;
}
"""


class HomeView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None

    def build(self) -> 'HomeView':
        with gr.Blocks(css=CSS) as self.app:
            with gr.Tab("Vector store"):
                VectorStoreView(app_context=self.app_context).build()

            with gr.Tab("Chatbot"):
                ChatbotView(app_context=self.app_context).build()

            with gr.Tab("Ingest"):
                IngestView(app_context=self.app_context).build()

            with gr.Tab("Settings"):
                SettingsView(app_context=self.app_context).build()

        return self


def main():
    app_context = AppContext.load_default()
    view = HomeView(app_context=app_context)
    view.build().app.launch(show_error=True)


if __name__ == '__main__':
    main()
