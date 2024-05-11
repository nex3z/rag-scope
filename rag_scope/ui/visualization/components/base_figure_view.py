from abc import ABC, abstractmethod
from typing import Optional, Callable, List

import gradio as gr
from gradio.events import Dependency
from loguru import logger

from rag_scope.retriever.base import Retrieval
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.visualizer.base_retrieval_visualizer import BaseRetrievalVisualizer


class BaseFigureView(BaseView, ABC):
    def __init__(
        self,
        app_context: AppContext,
        retrievals: gr.State,
        enabled: gr.State,
    ):
        self.app_context = app_context
        self.retrievals = retrievals
        self.enabled = enabled

        self.app: Optional[gr.Blocks] = None
        self.params: Optional[List] = None
        self.limit: Optional[gr.Number] = None
        self.seed: Optional[gr.Number] = None
        self.figure: Optional[gr.Plot] = None

    @abstractmethod
    def _build_params(self) -> List:
        pass

    @abstractmethod
    def _build_visualizer(self, seed: Optional[int], *params) -> BaseRetrievalVisualizer:
        pass

    def build(self) -> 'BaseFigureView':
        with gr.Blocks() as self.app:
            if self.enabled is None:
                self.enabled = gr.State(True)

            with gr.Accordion(label="Params"):
                with gr.Group():
                    self.params = self._build_params()

                    with gr.Row():
                        self.limit = gr.Number(
                            label="Limit",
                            value=500, minimum=-1, maximum=1000, step=1, interactive=True)
                        self.seed = gr.Number(
                            label="Seed",
                            value=-1, step=1, interactive=True)

            self.figure = gr.Plot(label="Figure")

            gr.on(
                triggers=[p.change for p in self.params] + [self.limit.change],
                fn=self.__on_plot,
                inputs=[self.enabled, self.retrievals, self.limit, self.seed, *self.params],
                outputs=[self.figure],
            )

        return self

    def update_on(self, event: Callable) -> Dependency:
        dep = event(
            self.__on_plot,
            [self.enabled, self.retrievals, self.limit, self.seed, *self.params],
            [self.figure]
        )
        return dep

    def __on_plot(
        self,
        enabled: bool,
        retrievals: Optional[List[Retrieval]],
        limit: int,
        seed: int,
        *params
    ):
        limit = None if limit == -1 else limit
        seed = None if seed == -1 else seed
        logger.info(f"enabled = {enabled}, limit = {limit}, seed = {seed}")

        if enabled is False:
            return
        elif retrievals is None:
            logger.warning("No retrievals, skip plot.")
            return
        elif len(retrievals) == 0:
            gr.Info("Nothing to visualize.")
            return
        # elif 'embedding' not in retrievals[0].documents[0].metadata:
        #     logger.error(f"Missing embedding, metadata = {retrievals.documents[0].metadata}")
        #     raise gr.Error("Missing embedding field in document metadata, cannot visualize.")

        try:
            visualizer = self._build_visualizer(seed, *params)
            figure = visualizer.render(retrievals, limit=limit)
        except Exception as e:
            raise gr.Error(f"Failed to visualize, error: {e}")

        return {
            self.figure: figure
        }
