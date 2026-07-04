"""Textual messages carrying training events from the worker to the UI thread."""

from __future__ import annotations

from textual.message import Message

from pureml.tui.trainers import Progress, Result


class TrainingStarted(Message):
    def __init__(self, model: str, dataset: str, total: int, unit: str) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.total = total
        self.unit = unit


class TrainingProgress(Message):
    def __init__(self, progress: Progress) -> None:
        super().__init__()
        self.progress = progress


class TrainingFinished(Message):
    def __init__(self, result: Result) -> None:
        super().__init__()
        self.result = result


class TrainingError(Message):
    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error
