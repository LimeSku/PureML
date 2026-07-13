import numpy as np


class Adam:
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, parameters_and_gradients) -> None:
        self.t += 1
        for param, grad in parameters_and_gradients:
            param_id = id(param)
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (
                grad**2
            )
            m_hat = self.m[param_id] / (1 - self.beta1**self.t)
            v_hat = self.v[param_id] / (1 - self.beta2**self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
