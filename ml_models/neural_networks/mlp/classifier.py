import numpy as np

from ml_models.neural_networks.mlp.activations import ReLU
from ml_models.neural_networks.mlp.layers import DenseLayer
from ml_models.neural_networks.mlp.losses import SoftmaxCrossEntropy


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        init_std: float = 0.01,
    ):
        self.layers = []
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.layers.append(DenseLayer(dims[i], dims[i + 1], init_std=init_std))
            is_output_layer = i == len(dims) - 2
            if not is_output_layer:
                self.layers.append(ReLU())
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        batch_size: int | None = None,
        log_every: int | None = None,
    ) -> list[float]:
        losses = []
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            batch_losses = []
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                logits = self.forward(X_batch)
                loss = self.loss_fn(logits, y_batch)
                batch_losses.append(loss)

                # backprop
                grad = self.loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                for layer in self.layers:
                    if isinstance(layer, DenseLayer):
                        layer.step(learning_rate=learning_rate)

            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if log_every is not None and (
                epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1
            ):
                initial_loss = losses[0]
                improvement = initial_loss - epoch_loss
                log_str = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f} - improvement: {improvement:.6f}"
                if X_val is not None and y_val is not None:
                    val_prediction = self.predict(X_val)
                    val_accuracy = np.mean(val_prediction == y_val)
                    log_str += f" - val_acc: {val_accuracy:.6f}"
                print(log_str)

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
