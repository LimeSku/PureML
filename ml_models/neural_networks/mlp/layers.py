import numpy as np


class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int, init_std: float = 0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.normal(
            0.0,
            scale=init_std,
            size=(input_dim, output_dim),
        )
        self.b = np.zeros(output_dim)
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        dX = dout @ self.W.T
        return dX

    def step(self, learning_rate: float):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
