from abc import ABC, abstractmethod

import gradio as gr

from rag_scope.ui.base.base_view import BaseView


class BaseRetrieverView(BaseView, ABC):
    @property
    @abstractmethod
    def retriever(self) -> gr.State:
        pass
