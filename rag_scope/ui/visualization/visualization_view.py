from typing import Optional, Callable

import gradio as gr
from gradio.events import Dependency
from loguru import logger

from rag_scope.retriever.query_vector_store_retriever import QueryVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.visualization.components.pca_view import PcaView
from rag_scope.ui.visualization.components.tsne_view import TsneView
from rag_scope.ui.visualization.components.umap_view import UmapView

TAB_PCA = "PCA"
TAB_UMAP = "UMAP"
TAB_TSNE = "TSNE"


class VisualizationView(BaseView):
    def __init__(self, app_context: AppContext, query_result: gr.State):
        self.app_context = app_context
        self.query_result = query_result

        self.app: Optional[gr.Blocks] = None
        self.pca_enabled: Optional[gr.Blocks] = None
        self.umap_enabled: Optional[gr.Blocks] = None
        self.tsne_enabled: Optional[gr.Blocks] = None
        self.pca_view: Optional[PcaView] = None
        self.umap_view: Optional[UmapView] = None
        self.tsne_view: Optional[UmapView] = None

    def build(self) -> 'VisualizationView':
        with gr.Blocks() as self.app:
            self.pca_enabled = gr.State(True)
            self.umap_enabled = gr.State(False)
            self.tsne_enabled = gr.State(False)

            with gr.Tab(TAB_PCA) as pca_tab:
                self.pca_view = PcaView(
                    app_context=self.app_context, retrievals=self.query_result, enabled=self.pca_enabled)
                self.pca_view.build()
                pass

            with gr.Tab(TAB_UMAP) as umap_tab:
                self.umap_view = UmapView(
                    app_context=self.app_context, retrievals=self.query_result, enabled=self.umap_enabled)
                self.umap_view.build()

            with gr.Tab(TAB_TSNE) as tsne_tab:
                self.tsne_view = TsneView(
                    app_context=self.app_context, retrievals=self.query_result, enabled=self.tsne_enabled)
                self.tsne_view.build()

            dep = gr.on(
                triggers=[pca_tab.select, umap_tab.select, tsne_tab.select],
                fn=self.__on_tab_selected,
                inputs=[],
                outputs=[self.pca_enabled, self.umap_enabled, self.tsne_enabled],
            )
            dep = self.pca_view.update_on(dep.then)
            dep = self.umap_view.update_on(dep.then)
            dep = self.tsne_view.update_on(dep.then)

            # pca_tab.select(self.__on_pca_selected, [], [self.pca_enabled])
            # umap_tab.select(self.__on_umap_selected, [], [])

        return self

    def update_on(self, event: Callable) -> Dependency:
        dep = self.pca_view.update_on(event)
        dep = self.umap_view.update_on(dep.then)
        dep = self.tsne_view.update_on(dep.then)
        return dep

    def __on_tab_selected(self, evt: gr.SelectData):
        enabled = {
            self.pca_enabled: False,
            self.umap_enabled: False,
            self.tsne_enabled: False,
        }
        if evt.value == TAB_PCA:
            enabled[self.pca_enabled] = True
        elif evt.value == TAB_UMAP:
            enabled[self.umap_enabled] = True
        elif evt.value == TAB_TSNE:
            enabled[self.tsne_enabled] = True
        logger.info(f"enabled = {enabled}")
        return enabled


def main():
    app_context = AppContext.load_default()
    retriever = QueryVectorStoreRetriever(
        engine=app_context.engine,
    )
    retrievals = retriever.retrieve("What is the authors view on the early stages of a startup?")

    with gr.Blocks() as demo:
        view = VisualizationView(app_context, gr.State(retrievals))
        view.build()

    demo.launch()


if __name__ == '__main__':
    main()
