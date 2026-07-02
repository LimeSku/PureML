import numpy as np


class ReLU:
    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(0.0, X)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.X > 0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
