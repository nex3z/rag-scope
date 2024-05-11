from typing import List, Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext
from rag_scope.ui.visualization.components.base_figure_view import BaseFigureView
from rag_scope.visualizer.base_retrieval_visualizer import BaseRetrievalVisualizer
from rag_scope.visualizer.umap_visualizer import UmapVisualizer


class UmapView(BaseFigureView):
    def __init__(self, app_context: AppContext, retrievals: gr.State, enabled: gr.State):
        super().__init__(app_context=app_context, retrievals=retrievals, enabled=enabled)

    def _build_params(self) -> List:
        n_component = gr.Radio(label="N components", choices=[2, 3], value=2, interactive=True)
        return [n_component]

    def _build_visualizer(self, seed: Optional[int], *params) -> BaseRetrievalVisualizer:
        n_component, = params

        visualizer = UmapVisualizer(
            stored=self.app_context.get_stored_documents(),
            n_components=n_component,
            random_state=seed,
        )
        return visualizer
