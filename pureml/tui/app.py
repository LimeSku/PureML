"""PureML live training dashboard (Textual TUI)."""

from __future__ import annotations

import dataclasses
import time

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Static

from pureml.tui.messages import (
    TrainingError,
    TrainingFinished,
    TrainingProgress,
    TrainingStarted,
)
from pureml.tui.registry import load_dataset, make_trainer
from pureml.tui.trainers import Progress
from pureml.tui.widgets.results_view import ResultsView
from pureml.tui.widgets.sidebar import Sidebar
from pureml.tui.widgets.training_view import TrainingView


class PureMLApp(App):
    TITLE = "PureML — Live Training Dashboard"
    CSS = """
    Horizontal#body {
        height: 1fr;
    }
    Sidebar {
        width: 32;
        border-right: solid $accent;
        padding: 1 1;
    }
    .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    .sidebar-note {
        margin-bottom: 1;
    }
    #models {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }
    #datasets {
        height: auto;
        margin-bottom: 1;
    }
    #hyperparams {
        height: auto;
    }
    Sidebar Input {
        width: 100%;
        margin-bottom: 1;
        border-title-color: $foreground;
    }
    /* Tree and Input hardcode ansi_default (black) text in native-ANSI mode;
       force the theme foreground so they stay readable on the transparency. */
    Tree, Input {
        color: $foreground;
    }
    #train {
        dock: bottom;
        margin-top: 1;
        width: 100%;
    }
    Vertical#main {
        padding: 1 2;
    }
    TrainingView {
        height: 12;
        border-bottom: solid $panel;
        padding-bottom: 1;
    }
    #status {
        text-style: bold;
        height: 1;
    }
    #progress {
        margin-top: 1;
    }
    #loss-row {
        height: 1fr;
        margin-top: 1;
    }
    #loss-pane {
        width: 2fr;
        margin-right: 2;
    }
    #metric-pane {
        width: 1fr;
    }
    #loss-spark {
        height: 4;
    }
    #live-metrics {
        min-height: 4;
    }
    ResultsView {
        height: 1fr;
        padding-top: 1;
    }
    #final-metrics {
        margin-bottom: 1;
    }
    #confusion-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #confusion {
        height: 1fr;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._training = False

    def on_mount(self) -> None:
        # Native-ANSI theme: every background resolves to the terminal default,
        # so Ghostty's transparency/blur shows through the whole app. Clone it with
        # a light foreground so text stays readable on the dark, translucent
        # background (the stock theme uses ansi_default text, which falls back to
        # black). Colors otherwise come from the terminal's ANSI palette.
        base = self.available_themes["ansi-dark"]
        # The stock theme uses ansi_black for the (unfocused) border, which also
        # colors border titles like the hyperparameter labels -> invisible on the
        # dark transparency. Lift it to a readable grey.
        variables = {**base.variables, "border-blurred": "ansi_bright_black"}
        self.register_theme(
            dataclasses.replace(
                base,
                name="pureml-ansi",
                foreground="ansi_bright_white",
                variables=variables,
            )
        )
        self.theme = "pureml-ansi"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield Sidebar()
            with Vertical(id="main"):
                yield TrainingView()
                yield ResultsView()
        yield Footer()

    @on(Button.Pressed, "#train")
    def _on_train_pressed(self) -> None:
        if self._training:
            return
        sidebar = self.query_one(Sidebar)
        model = sidebar.selected_model()
        dataset = sidebar.selected_dataset()
        if not model or not dataset:
            return
        params = sidebar.hyperparams()
        max_rows = sidebar.max_rows()

        self._set_training(True)
        self.query_one("#status", Static).update(f"Loading {dataset} ...")
        self._train(model, dataset, params, max_rows)

    @work(thread=True, exclusive=True)
    def _train(
        self, model_name: str, dataset_name: str, params: dict, max_rows: int
    ) -> None:
        try:
            last_progress_post = 0.0

            def emit_progress(progress: Progress) -> None:
                nonlocal last_progress_post
                now = time.monotonic()
                is_done = progress.step >= progress.total
                has_loss = progress.loss is not None
                if has_loss or is_done or now - last_progress_post >= 0.1:
                    last_progress_post = now
                    self.post_message(TrainingProgress(progress))

            dataset = load_dataset(dataset_name, max_rows)
            trainer = make_trainer(model_name, params)
            self.post_message(
                TrainingStarted(model_name, dataset_name, trainer.total, trainer.unit)
            )
            result = trainer.run(dataset, emit_progress)
            self.post_message(TrainingFinished(result))
        except Exception as exc:  # surface any failure in the UI, never crash silently
            self.post_message(TrainingError(str(exc)))

    def on_training_started(self, message: TrainingStarted) -> None:
        self.query_one(TrainingView).start(
            message.model, message.dataset, message.total, message.unit
        )
        self.query_one(ResultsView).clear()

    def on_training_progress(self, message: TrainingProgress) -> None:
        self.query_one(TrainingView).update_progress(message.progress)

    def on_training_finished(self, message: TrainingFinished) -> None:
        self.query_one(ResultsView).show(message.result)
        self.query_one("#status", Static).update("Done.")
        self._set_training(False)

    def on_training_error(self, message: TrainingError) -> None:
        self.query_one(TrainingView).set_error(message.error)
        self._set_training(False)

    def _set_training(self, active: bool) -> None:
        self._training = active
        train = self.query_one("#train", Button)
        train.disabled = active
        train.label = "Training ..." if active else "Train selected model"


def main() -> None:
    PureMLApp().run()


if __name__ == "__main__":
    main()
