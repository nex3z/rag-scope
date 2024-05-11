import json
from typing import Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext, AVAILABLE_EMBEDDINGS_CLASSES
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.utils.message_utils import on_load_success


class EmbeddingsSettingsView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self.status: Optional[gr.Textbox] = None
        self.embeddings_class: Optional[gr.Dropdown] = None
        self.embeddings_param: Optional[gr.Code] = None
        self.query: Optional[gr.Textbox] = None
        self.embedding: Optional[gr.Textbox] = None

    def build(self) -> 'EmbeddingsSettingsView':
        with gr.Blocks() as self.app:
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        self.status = gr.Textbox(
                            label="Embeddings status",
                            lines=1, max_lines=1, interactive=False)

                    self.embeddings_class = gr.Dropdown(
                        label="Embeddings class",
                        choices=AVAILABLE_EMBEDDINGS_CLASSES,
                        filterable=True, allow_custom_value=True, interactive=True)

                    self.embeddings_param = gr.Code(
                        label="Embeddings param",
                        lines=10, language='json', interactive=True)

                    btn_load = gr.Button("Load embeddings", variant='primary')
                    btn_load.click(
                        self.__on_load_embeddings,
                        [self.embeddings_class, self.embeddings_param],
                        [self.status]
                    ).success(on_load_success)

                with gr.Column(variant='panel', scale=1):
                    with gr.Row():
                        self.query = gr.Textbox(label="Query", scale=7, interactive=True)

                        with gr.Column(min_width=32):
                            btn_embed = gr.Button("Embed", variant='primary')
                            btn_clean = gr.ClearButton()

                    with gr.Row():
                        self.embedding = gr.DataFrame(type='array', interactive=False)

                    btn_embed.click(
                        self.__on_embed_query,
                        [self.query],
                        [self.embedding]
                    )
                    btn_clean.add([self.query, self.embedding])

            self.app.load(
                self.__on_load,
                [],
                [self.status, self.embeddings_class, self.embeddings_param]
            )

        return self

    def __on_load(self):
        return {
            self.status: self.__build_status(),
            self.embeddings_class: self.app_context.embeddings_class,
            self.embeddings_param: json.dumps(self.app_context.embeddings_param, indent=2),
        }

    def __build_status(self) -> str:
        if self.app_context.engine.embeddings is not None:
            return "Ready"
        else:
            return "Not ready"

    def __on_load_embeddings(self, embeddings_class: str, embeddings_param: str):
        embeddings_param = json.loads(embeddings_param)
        self.app_context.load_embeddings(embeddings_class, embeddings_param)
        return {
            self.status: self.__build_status(),
        }

    def __on_embed_query(self, query: str):
        embedding = self.app_context.engine.embeddings.embed_query(query)
        return {
            self.embedding: [embedding],
        }


def main():
    app_context = AppContext()
    view = EmbeddingsSettingsView(app_context).build()
    view.app.launch()


if __name__ == '__main__':
    main()
