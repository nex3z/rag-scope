import typer

from rag_scope.scripts.build_vector_store import build_vector_store
from rag_scope.ui.app_context import AppContext
from rag_scope.ui.home_view import HomeView

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command(help="Show web UI。")
def launch(
    default: bool = typer.Option(False, '--default',  help="Load default config."),
):
    if default is True:
        app_context = AppContext.load_default()
    else:
        app_context = AppContext()

    view = HomeView(app_context=app_context)
    view.build().app.launch()


@app.command(help="Show demo web UI。")
def launch_demo(
    init_vector_store: bool = typer.Option(False, '--init',  help="Load default config."),
):
    if init_vector_store is True:
        build_vector_store(reset=True)

    app_context = AppContext.load_default()
    view = HomeView(app_context=app_context)
    view.build().app.launch()


if __name__ == "__main__":
    app()
