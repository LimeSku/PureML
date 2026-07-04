"""Final-results panel: scalar metrics + a confusion-matrix table."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from pureml.tui.trainers import Result


class ResultsView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("No completed run yet.", id="final-metrics")
        yield Static("Confusion matrix", id="confusion-title")
        yield DataTable(id="confusion", zebra_stripes=True)

    def clear(self) -> None:
        self.query_one("#final-metrics", Static).update("Training in progress ...")
        self.query_one("#confusion", DataTable).clear(columns=True)

    def show(self, result: Result) -> None:
        lines = [
            "[b]Results[/b]",
            f"Train accuracy  [b]{result.train_acc:.3f}[/b]",
            f"Test accuracy   [b]{result.test_acc:.3f}[/b]",
            f"Elapsed         {result.elapsed:.2f}s",
        ]
        if result.oob_score is not None:
            lines.append(f"OOB score       {result.oob_score:.3f}")
        self.query_one("#final-metrics", Static).update("\n".join(lines))

        self._render_confusion(result)

    def _render_confusion(self, result: Result) -> None:
        table = self.query_one("#confusion", DataTable)
        table.clear(columns=True)
        table.cursor_type = "none"
        if result.confusion is None:
            return

        names = result.class_names
        table.add_column("true \\ pred")
        for name in names:
            table.add_column(name)
        for row_index, name in enumerate(names):
            row = result.confusion[row_index]
            table.add_row(name, *(str(int(count)) for count in row))
