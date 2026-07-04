"""Left-hand navigation: pick a model + dataset, tune hyperparameters, then Train."""

from __future__ import annotations

from collections.abc import Iterator

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Tree
from textual.widgets.tree import TreeNode

from pureml.tui.registry import (
    DATASETS,
    DEFAULT_MAX_ROWS,
    MODEL_FAMILIES,
    HyperParam,
    model_params,
)


class Sidebar(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Label("Configure a run", classes="sidebar-note")

        yield Label("Models", classes="section-title")
        yield self._build_model_tree()

        yield Label("Datasets", classes="section-title")
        with RadioSet(id="datasets"):
            for index, name in enumerate(DATASETS):
                yield RadioButton(name, value=index == 0)

        yield Label("Data", classes="section-title")
        max_rows = Input(
            value=str(DEFAULT_MAX_ROWS), type="integer", id="max-rows"
        )
        max_rows.border_title = "Max train rows"
        yield max_rows

        yield Label("Hyperparameters", classes="section-title")
        with Vertical(id="hyperparams"):
            yield from self._hyperparam_inputs(self._selected_model)

        yield Button("Train selected model", id="train", variant="success")

    # -- model tree ---------------------------------------------------------

    def _build_model_tree(self) -> Tree:
        tree: Tree[str] = Tree("Models", id="models")
        tree.show_root = False
        tree.guide_depth = 3
        self._model_nodes: dict[str, TreeNode] = {}
        self._selected_model: str | None = None
        for family, model_names in MODEL_FAMILIES.items():
            family_node = tree.root.add(family, expand=True)
            for name in model_names:
                # Leaf nodes carry the model name in ``data`` for selection.
                node = family_node.add_leaf(name, data=name)
                self._model_nodes[name] = node
                if self._selected_model is None:
                    self._selected_model = name
        self._refresh_model_markers()
        return tree

    def _refresh_model_markers(self) -> None:
        """Show a dot next to the currently selected model leaf."""
        for name, node in self._model_nodes.items():
            selected = name == self._selected_model
            label = Text()
            label.append("● " if selected else "  ", style="bold green")
            label.append(name)
            node.set_label(label)

    async def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        # Family nodes have data=None; only react to an actual model leaf.
        if event.node.data is None or event.node.data == self._selected_model:
            return
        self._selected_model = event.node.data
        self._refresh_model_markers()
        # Swap the hyperparameter inputs for the newly selected model.
        container = self.query_one("#hyperparams", Vertical)
        await container.remove_children()
        await container.mount(*self._hyperparam_inputs(self._selected_model))

    # -- hyperparameter inputs ---------------------------------------------

    def _hyperparam_inputs(self, model_name: str) -> Iterator[Input]:
        self._current_params: list[HyperParam] = model_params(model_name)
        for param in self._current_params:
            value = str(int(param.default)) if param.is_int else str(param.default)
            field = Input(
                value=value,
                type="integer" if param.is_int else "number",
                id=f"hp-{param.key}",
            )
            field.border_title = param.label
            yield field

    # -- read selections ----------------------------------------------------

    def selected_model(self) -> str:
        return self._selected_model or ""

    def selected_dataset(self) -> str:
        radio_set = self.query_one("#datasets", RadioSet)
        button = radio_set.pressed_button
        return str(button.label) if button is not None else ""

    def max_rows(self) -> int:
        return self._read_int("#max-rows", DEFAULT_MAX_ROWS)

    def hyperparams(self) -> dict:
        values: dict = {}
        for param in self._current_params:
            field = self.query_one(f"#hp-{param.key}", Input)
            raw = field.value.strip()
            try:
                values[param.key] = int(raw) if param.is_int else float(raw)
            except ValueError:
                values[param.key] = param.default
        return values

    def _read_int(self, selector: str, fallback: int) -> int:
        raw = self.query_one(selector, Input).value.strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return fallback
