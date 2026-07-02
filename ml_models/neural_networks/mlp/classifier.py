import numpy as np

from ml_models.neural_networks.mlp.activations import ReLU
from ml_models.neural_networks.mlp.layers import DenseLayer
from ml_models.neural_networks.mlp.losses import SoftmaxCrossEntropy


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        init_std: float = 0.01,
    ):
        self.dense1 = DenseLayer(input_dim, hidden_dim, init_std=init_std)
        self.activation = ReLU()
        self.dense2 = DenseLayer(hidden_dim, num_classes, init_std=init_std)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = self.dense1(X)
        out = self.activation(out)
        out = self.dense2(out)
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
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
                dlogits = self.loss_fn.backward()
                dout = self.dense2.backward(dlogits)
                dout2 = self.activation.backward(dout)
                self.dense1.backward(dout2)
                self.dense1.step(learning_rate)
                self.dense2.step(learning_rate)
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if log_every is not None and (
                epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1
            ):
                initial_loss = losses[0]
                improvement = initial_loss - epoch_loss
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- loss: {epoch_loss:.6f} "
                    f"- improvement: {improvement:.6f}"
                )
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
