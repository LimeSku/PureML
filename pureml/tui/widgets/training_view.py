"""Live training panel: status header, progress bar, loss sparkline, metrics."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, ProgressBar, Sparkline, Static

from pureml.tui.trainers import Progress


class TrainingView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("Select a model and dataset, then press Train.", id="status")
        yield ProgressBar(id="progress", total=100, show_eta=False)
        with Horizontal(id="loss-row"):
            with Vertical(id="loss-pane"):
                yield Label("Loss", classes="section-title", id="loss-label")
                yield Sparkline([], id="loss-spark")
            with Vertical(id="metric-pane"):
                yield Label("Run", classes="section-title")
                yield Static("Idle", id="live-metrics")

    def start(self, model: str, dataset: str, total: int, unit: str) -> None:
        self._losses = []
        self.query_one("#status", Static).update(
            f"Training [b]{model}[/b] on [b]{dataset}[/b] ..."
        )
        progress = self.query_one("#progress", ProgressBar)
        progress.update(total=total, progress=0)
        self.query_one("#loss-spark", Sparkline).data = []
        self.query_one("#live-metrics", Static).update(f"{unit.title()}\n0/{total}")

    def update_progress(self, progress: Progress) -> None:
        bar = self.query_one("#progress", ProgressBar)
        bar.update(total=progress.total, progress=progress.step)

        parts = [
            progress.unit.title(),
            f"{progress.step}/{progress.total}",
        ]
        if progress.loss is not None:
            parts.append(f"Loss {progress.loss:.4f}")
            self._losses.append(progress.loss)
            self.query_one("#loss-spark", Sparkline).data = self._losses
        else:
            parts.append("Loss unavailable")
        if progress.train_acc is not None:
            parts.append(f"Train acc {progress.train_acc:.3f}")
        self.query_one("#live-metrics", Static).update("\n".join(parts))

    def set_error(self, message: str) -> None:
        self.query_one("#status", Static).update(f"[b red]Error:[/b red] {message}")
