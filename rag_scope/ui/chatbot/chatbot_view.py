from itertools import chain
from typing import Optional, List

import gradio as gr
import pandas as pd
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger

from rag_scope.retriever.base import Retrieval
from rag_scope.retriever.base_vector_store_retriever import BaseVectorStoreRetriever
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.retriever.retriever_select_view import RetrieverSelectView
from rag_scope.ui.visualization.visualization_view import VisualizationView
from rag_scope.utils.document_utils import format_documents_to_string, format_retrievals_to_data_frame

DEFAULT_SYSTEM_TEMPLATE = """You are a helpful assistant."""
DEFAULT_USER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#row_prompt_bot { flex-grow: 1; overflow: auto;}
#chatbot { flex-grow: 1; overflow: auto;}
footer {visibility: hidden}
"""


class ChatbotView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self.retrievals: Optional[gr.State] = None

        self.query: Optional[gr.Textbox] = None
        self.chatbot: Optional[gr.Chatbot] = None
        self.documents: Optional[gr.DataFrame] = None

    def build(self) -> 'ChatbotView':
        with gr.Blocks() as self.app:
            self.retrievals = gr.State()

            with gr.Row(equal_height=True, elem_id='row_prompt_bot'):
                with gr.Column(scale=1):
                    with gr.Accordion(label="Prompt"):
                        with gr.Group():
                            system_prompt = gr.Code(
                                DEFAULT_SYSTEM_TEMPLATE,
                                interactive=True,
                                lines=5, label="System",
                            )
                            user_prompt = gr.Code(
                                DEFAULT_USER_TEMPLATE,
                                interactive=True,
                                lines=5, label="User",
                            )
                            enable_history = gr.Checkbox(
                                value=False,
                                label="Enable history",
                            )

                    with gr.Accordion(label="Retriever"):
                        retriever_view = RetrieverSelectView(self.app_context)
                        retriever_view.build()

                with gr.Column(scale=1):
                    self.chatbot = gr.Chatbot(
                        [],
                        elem_id='chatbot',
                        height='86vh',
                        bubble_full_width=False,

                    )

                    with gr.Row():
                        self.query = gr.Textbox(
                            scale=5,
                            show_label=False,
                            placeholder="",
                            container=False,
                            interactive=True,
                        )

                        gr.ClearButton(
                            [self.query, self.chatbot],
                            variant='stop', value="🗑", scale=0, min_width=48)

                with gr.Column(scale=1):
                    with gr.Accordion(label="Visualization"):
                        viz_view = VisualizationView(app_context=self.app_context, query_result=self.retrievals)
                        viz_view.build()

                    with gr.Accordion(label="Documents"):
                        self.documents = gr.DataFrame(
                            pd.DataFrame({'score': [None], 'page_content': [""]}),
                            wrap=True)

                dep = self.query.submit(
                    self.__on_query,
                    [retriever_view.retriever, self.query, self.chatbot],
                    [self.query, self.retrievals, self.documents, self.chatbot]
                )
                dep = viz_view.update_on(dep.success)
                dep.success(
                    self.__on_chat,
                    [system_prompt, user_prompt, self.retrievals, self.chatbot, enable_history],
                    [self.chatbot]
                )

        return self

    def __on_query(self, retriever: BaseVectorStoreRetriever, query: str, history: List[List[str]]):
        retrievals = retriever.retrieve(query)
        df_documents = format_retrievals_to_data_frame(retrievals)
        history = history + [[retrievals[0].query.query_content, None]]
        return {
            self.query: "",
            self.retrievals: retrievals,
            self.documents: df_documents,
            self.chatbot: history
        }

    def __on_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        retrievals: List[Retrieval],
        history: List[List[str]],
        enable_history: bool
    ):
        messages = []
        if system_prompt is not None and len(system_prompt) != 0:
            messages.append(SystemMessage(content=system_prompt))
        if enable_history is True:
            messages.extend(format_messages(history[:-1]))
        documents = list(chain.from_iterable(r.documents for r in retrievals))
        prompt = PromptTemplate.from_template(
            user_prompt,
            partial_variables={'context': format_documents_to_string(documents)}
        )
        message = HumanMessage(content=prompt.format(question=retrievals[0].query.query_content))
        messages.append(message)
        logger.info(f"messages = {messages}")
        response = self.app_context.stream(messages)
        history[-1][1] = ""

        for char in response:
            history[-1][1] += char
            yield {
                self.chatbot: history
            }


def format_messages(messages: List[List[str]]) -> List[BaseMessage]:
    formatted = []
    for pair in messages:
        formatted.append(HumanMessage(content=pair[0]))
        formatted.append(AIMessage(content=pair[1]))
    return formatted


def main():
    app_context = AppContext.load_default()
    view = ChatbotView(app_context=app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
