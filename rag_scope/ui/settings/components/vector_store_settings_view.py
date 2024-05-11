import json
from typing import Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext, AVAILABLE_VECTOR_STORE_CLASSES
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.utils.message_utils import on_load_success
from rag_scope.utils.document_utils import format_documents_to_data_frame


class VectorStoreSettingsView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.status: Optional[gr.Textbox] = None
        self.vector_store_class: Optional[gr.Dropdown] = None
        self.vector_store_param: Optional[gr.Code] = None
        self.query: Optional[gr.Textbox] = None
        self.documents: Optional[gr.DataFrame] = None

    def build(self) -> 'VectorStoreSettingsView':
        with gr.Blocks() as self.app:
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        self.status = gr.Textbox(
                            label="Vector store status",
                            max_lines=1, interactive=False)

                    self.vector_store_class = gr.Dropdown(
                        label="Vector store class",
                        choices=AVAILABLE_VECTOR_STORE_CLASSES,
                        filterable=True, allow_custom_value=True, interactive=True)

                    self.vector_store_param = gr.Code(
                        label="Vector store params",
                        lines=10, language='json', interactive=True)

                    btn_load = gr.Button("Load vector store", variant='primary')
                    btn_load.click(
                        self.__on_load_vector_store,
                        [self.vector_store_class, self.vector_store_param],
                        [self.status]
                    ).success(on_load_success)

                with gr.Column(variant='panel', scale=1):
                    with gr.Row():
                        self.query = gr.Textbox(label="Query", scale=7, interactive=True)

                        with gr.Column(min_width=32):
                            btn_query = gr.Button("Search", variant='primary')
                            btn_clean = gr.ClearButton()

                    with gr.Row():
                        self.documents = gr.DataFrame(wrap=True)

                    btn_query.click(
                        self.__on_query,
                        [self.query],
                        [self.documents]
                    )
                    btn_clean.add([self.query, self.documents])

            self.app.load(
                self.__on_load,
                [],
                [self.status, self.vector_store_class, self.vector_store_param]
            )

        return self

    def __on_load(self):
        return {
            self.status: self.__build_status(),
            self.vector_store_class: self.app_context.vector_store_class,
            self.vector_store_param: json.dumps(self.app_context.vector_store_param, indent=2),
        }

    def __build_status(self) -> str:
        if self.app_context.engine.vector_store is not None:
            return "Ready"
        else:
            return "Not ready"

    def __on_load_vector_store(self, vector_store_class: str, vector_store_param: str):
        vector_store_param = json.loads(vector_store_param)
        self.app_context.load_vector_store(vector_store_class, vector_store_param)
        return {
            self.status: self.__build_status(),
        }

    def __on_query(self, query: str):
        embedding = self.app_context.engine.embeddings.embed_query(query)
        documents = self.app_context.engine.vector_store.similarity_search_by_vector(embedding)
        df = format_documents_to_data_frame(documents)
        return {
            self.documents: df
        }


def main():
    app_context = AppContext()
    app_context.load_embeddings()
    view = VectorStoreSettingsView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
