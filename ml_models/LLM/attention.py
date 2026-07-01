import numpy as np


class CausalSelfAttentionHead:
    def __init__(self, embedding_dim: int, head_dim: int, init_std: float = 0.02):
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.W_query = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))
        self.W_key = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))
        self.W_value = np.random.normal(0.0, init_std, size=(embedding_dim, head_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value

        attention_scores = Q @ K.T
        attention_scores = attention_scores / np.sqrt(self.head_dim)

        T = x.shape[0]  # context length
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        attention_scores[mask] = -np.inf
        attention_weights = self._softmax(attention_scores)

        return attention_weights @ V

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MultiHeadCausalSelfAttention:
    def __init__(self, embedding_dim: int, num_heads: int, init_std: float = 0.02):
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.heads = [
            CausalSelfAttentionHead(
                embedding_dim=embedding_dim, head_dim=self.head_dim, init_std=init_std
            )
            for _ in range(num_heads)
        ]
        self.W_output = np.random.normal(
            0.0,
            init_std,
            size=(embedding_dim, embedding_dim),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        head_outputs = [head(x) for head in self.heads]
        concatenated = np.concatenate(head_outputs, axis=1)
        return concatenated @ self.W_output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
