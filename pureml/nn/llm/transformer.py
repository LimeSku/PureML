import numpy as np

from pureml.nn.llm.attention import MultiHeadCausalSelfAttention


class FeedForward:
    def __init__(self, embedding_dim: int, hidden_dim: int, init_std: float = 0.02):
        self.W1 = np.random.normal(
            0.0,
            scale=init_std,
            size=(embedding_dim, hidden_dim),
        )
        # bias term
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.normal(
            0.0,
            scale=init_std,
            size=(hidden_dim, embedding_dim),
        )
        self.b2 = np.zeros(embedding_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.hidden_pre_activation = x @ self.W1 + self.b1
        self.hidden = self._gelu(self.hidden_pre_activation)
        return self.hidden @ self.W2 + self.b2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW2 = self.hidden.T @ dout
        self.db2 = np.sum(dout, axis=0)
        dhidden = dout @ self.W2.T
        dhidden_pre_activation = dhidden * self._gelu_derivative(
            self.hidden_pre_activation
        )
        self.dW1 = self.x.T @ dhidden_pre_activation
        self.db1 = np.sum(dhidden_pre_activation, axis=0)
        dx = dhidden_pre_activation @ self.W1.T
        return dx

    def step(self, learning_rate: float) -> None:
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        # :)
        # Gaussian Error linear unit, makes the transformation not just a linear projection.
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_derivative(self, x):
        """copy pasted dgelu ... no shame"""
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)

        return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * c * (
            1.0 + 3.0 * 0.044715 * x**2
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters_and_gradients(self):
        return [
            (self.W1, self.dW1),
            (self.b1, self.db1),
            (self.W2, self.dW2),
            (self.b2, self.db2),
        ]


class LayerNorm:
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(self.variance + self.eps)
        self.x_hat = (x - self.mean) * self.inv_std
        return self.x_hat * self.gamma + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        n_features = dout.shape[-1]  # last dimension = features
        dx_hat = dout * self.gamma
        dx = (
            self.inv_std
            / n_features
            * (
                n_features * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
            )
        )
        return dx

    def step(self, learning_rate: float) -> None:
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta

    def parameters_and_gradients(self):
        return [
            (self.gamma, self.dgamma),
            (self.beta, self.dbeta),
        ]


class TransformerBlock:
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ):
        self.ln1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, init_std=init_std
        )

        self.ln2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, init_std=init_std
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.attention_residual = x
        x = x + self.attention(self.ln1(x))
        self.feed_forward_residual = x
        x = x + self.feed_forward(self.ln2(x))
        return x

    def backward(self, dout):
        # xout = feedforward residual + feedforward(ln2(feedforwardresidual))
        dresidual = dout
        dff = dout

        dln2 = self.feed_forward.backward(dff)
        dff_input = self.ln2.backward(dln2)
        dx = dresidual + dff_input

        # first residual branch:
        # feed forward residual = attention residual + attention(ln1(attention_residual))
        dresidual = dx
        dattention = dx
        dln1 = self.attention.backward(dattention)
        dattention_input = self.ln1.backward(dln1)
        dx = dresidual + dattention_input
        return dx

    def step(self, learning_rate: float):
        self.ln1.step(learning_rate)
        self.attention.step(learning_rate)
        self.ln2.step(learning_rate)
        self.feed_forward.step(learning_rate)

    def parameters_and_gradients(self):
        return (
            self.ln1.parameters_and_gradients()
            + self.attention.parameters_and_gradients()
            + self.ln2.parameters_and_gradients()
            + self.feed_forward.parameters_and_gradients()
        )
