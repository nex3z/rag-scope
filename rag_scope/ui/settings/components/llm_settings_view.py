import json
from typing import Optional

import gradio as gr

from rag_scope.ui.app_context import AppContext, AVAILABLE_CHAT_MODELS
from rag_scope.ui.base.base_view import BaseView
from rag_scope.ui.utils.message_utils import on_load_success


class LlmSettingsView(BaseView):
    def __init__(self, app_context: AppContext):
        self.app_context = app_context

        self.app: Optional[gr.Blocks] = None
        self.status: Optional[gr.Textbox] = None
        self.chat_model_class: Optional[gr.Dropdown] = None
        self.chat_model_param: Optional[gr.Code] = None
        self.user_input: Optional[gr.Textbox] = None
        self.response: Optional[gr.Textbox] = None

    def build(self) -> 'LlmSettingsView':
        with gr.Blocks() as self.app:
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        self.status = gr.Textbox(
                            label="Chat model status",
                            lines=1, max_lines=1, interactive=False)

                    self.chat_model_class = gr.Dropdown(
                        label="Chat model class",
                        choices=AVAILABLE_CHAT_MODELS,
                        filterable=True, allow_custom_value=True, interactive=True)

                    self.chat_model_param = gr.Code(
                        label="Chat model param",
                        lines=10, language='json', interactive=True)

                    btn_load = gr.Button("Load chat model", variant='primary')
                    btn_load.click(
                        self.__on_load_chat_model,
                        [self.chat_model_class, self.chat_model_param],
                        [self.status]
                    ).success(on_load_success)

                with gr.Column(variant='panel', scale=1):
                    with gr.Row():
                        self.user_input = gr.Textbox(label="Input", scale=7, interactive=True)

                        with gr.Column(min_width=32):
                            btn_chat = gr.Button("Submit", variant='primary')
                            btn_clean = gr.ClearButton()

                    with gr.Row():
                        self.response = gr.Text(label="Response", lines=10)

                    gr.on(
                        triggers=[btn_chat.click, self.user_input.submit],
                        fn=self.__on_chat,
                        inputs=[self.user_input],
                        outputs=[self.response]
                    )

                    btn_clean.add([self.user_input, self.response])

            self.app.load(
                self.__on_load,
                [],
                [self.status, self.chat_model_class, self.chat_model_param],
            )

        return self

    def __on_load(self):
        return {
            self.status: self.__build_status(),
            self.chat_model_class: self.app_context.chat_model_class,
            self.chat_model_param: json.dumps(self.app_context.chat_model_param, indent=2),
        }

    def __build_status(self) -> str:
        if self.app_context.engine.chat_model is not None:
            return "Ready"
        else:
            return "Not ready"

    def __on_load_chat_model(self, chat_model_class: str, chat_model_param: str):
        chat_model_param = json.loads(chat_model_param)
        self.app_context.load_chat_model(chat_model_class, chat_model_param)
        return {
            self.status: self.__build_status(),
        }

    def __on_chat(self, user_input: str):
        response = self.app_context.chat(user_input)
        return {
            self.response: response
        }


def main():
    app_context = AppContext()
    view = LlmSettingsView(app_context)
    view.build().app.launch()


if __name__ == '__main__':
    main()
