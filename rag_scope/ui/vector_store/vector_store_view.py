from typing import Optional, List

import gradio as gr
import pandas as pd

from rag_scope.retriever.base import Retrieval
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.retriever.vector_store_retriever_view import VectorStoreRetrieverView
from rag_scope.ui.visualization.visualization_view import VisualizationView
from rag_scope.utils.document_utils import format_retrievals_to_data_frame


class VectorStoreView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self.retrievals: Optional[gr.State] = None

        self.documents: Optional[gr.DataFrame] = None
        self.figure: Optional[gr.Plot] = None

    def build(self) -> 'VectorStoreView':
        with gr.Blocks() as self.app:
            self.retrievals = gr.State()

            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        query = gr.Textbox(label="Query", scale=7, interactive=True)

                        with gr.Column(min_width=32):
                            btn_query = gr.Button("Query", variant='primary')
                            btn_clear = gr.ClearButton()

                    with gr.Accordion(label="Search settings"):
                        retriever_view = VectorStoreRetrieverView(self.app_context)
                        retriever_view.build()

                    self.documents = gr.DataFrame(
                        pd.DataFrame({'score': [None], 'page_content': [""]}),
                        wrap=True)

                viz_view = VisualizationView(self.app_context, query_result=self.retrievals)
                viz_view.build()

                btn_clear.add([
                    query, self.documents,
                    viz_view.pca_view.figure, viz_view.umap_view.figure, viz_view.tsne_view.figure
                ])

                dep = gr.on(
                    triggers=[btn_query.click, query.submit],
                    fn=self.__on_query,
                    inputs=[retriever_view.retriever, query],
                    outputs=[self.retrievals]
                ).success(
                    self.__render_documents,
                    [self.retrievals],
                    [self.documents]
                )

                viz_view.update_on(dep.success)

        return self

    def __on_query(self, retriever: BaseVectorStoreRetriever, query: str):
        retrievals = retriever.retrieve(query)
        return {
            self.retrievals: retrievals,
        }

    def __render_documents(self, retrievals: List[Retrieval]):
        df = format_retrievals_to_data_frame(retrievals)
        return {
            self.documents: df
        }


def main():
    app_context = AppContext()
    app_context.load_embeddings()
    app_context.load_vector_store()
    view = VectorStoreView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
