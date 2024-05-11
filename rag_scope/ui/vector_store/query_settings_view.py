from typing import Optional

import gradio as gr
from pydantic import BaseModel

from rag_scope.embeddings.embed_type import EmbedType
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.vector_store.search_type import SearchType


class SearchSettings(BaseModel):
    embed_type: EmbedType = EmbedType.QUERY
    search_type: SearchType = SearchType.SIMILARITY
    k: int = 4


class QuerySettingsView(BaseView):
    def __init__(self, app_context: AppContext, default: Optional[SearchSettings] = None):
        self.app_context = app_context
        self.default = default if default is not None else SearchSettings()

        self.app: Optional[gr.Blocks] = None
        self.query_settings: Optional[gr.State] = None

    def build(self) -> 'QuerySettingsView':
        with gr.Blocks() as self.app:
            self.query_settings = gr.State(self.default)

            with gr.Row():
                # noinspection PyUnresolvedReferences
                embed_type = gr.Radio(
                    label="Embed type",
                    choices=[e.value for e in EmbedType],
                    value=self.default.embed_type,
                    interactive=True,
                )

                # noinspection PyUnresolvedReferences
                query_method = gr.Radio(
                    label="Search type",
                    choices=[e.value for e in SearchType],
                    value=self.default.search_type,
                    interactive=True,
                )

            k = gr.Slider(
                label="k",
                minimum=1, maximum=100, step=1,
                value=self.default.k,
                interactive=True
            )

            gr.on(
                triggers=[embed_type.change, query_method.change, k.change],
                fn=self.__on_change,
                inputs=[embed_type, query_method, k],
                outputs=[self.query_settings]
            )

        return self

    def __on_change(self, embed_type: str, search_type: str, k: int):
        query_settings = SearchSettings(
            embed_type=EmbedType(embed_type),
            search_type=SearchType(search_type),
            k=k,
        )
        return {
            self.query_settings: query_settings
        }


def main():
    app_context = AppContext()
    with gr.Blocks() as demo:
        view = QuerySettingsView(app_context)
        view.build()
    demo.launch()


if __name__ == '__main__':
    main()
