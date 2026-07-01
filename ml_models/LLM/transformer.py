import numpy as np

from ml_models.LLM.attention import MultiHeadCausalSelfAttention


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
        hidden = x @ self.W1 + self.b1
        hidden = self._gelu(hidden)
        return hidden @ self.W2 + self.b2

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        # :)
        # Gaussian Error linear unit, makes the transformation not just a linear projection.
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class LayerNorm:
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return normalized * self.gamma + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
