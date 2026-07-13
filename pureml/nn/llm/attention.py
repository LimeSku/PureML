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

        self.x = x
        self.Q = Q
        self.K = K
        self.V = V
        self.scores = attention_scores
        self.mask = mask
        self.weights = attention_weights
        return attention_weights @ V

    def backward(self, dout):
        dweights = dout @ self.V.T
        dV = self.weights.T @ dout
        dscores = self.weights * (
            dweights - np.sum(dweights * self.weights, axis=-1, keepdims=True)
        )
        dscores[self.mask] = 0.0
        dscores = dscores / np.sqrt(self.head_dim)
        dQ = dscores @ self.K
        dK = dscores.T @ self.Q

        self.dW_query = self.x.T @ dQ
        self.dW_key = self.x.T @ dK
        self.dW_value = self.x.T @ dV

        # dX = sum(gradients from all paths)
        dx = dQ @ self.W_query.T + dK @ self.W_key.T + dV @ self.W_value.T

        return dx

    def step(self, learning_rate: float) -> None:
        self.W_query -= learning_rate * self.dW_query
        self.W_key -= learning_rate * self.dW_key
        self.W_value -= learning_rate * self.dW_value

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def parameters_and_gradients(self):
        return [
            (self.W_query, self.dW_query),
            (self.W_key, self.dW_key),
            (self.W_value, self.dW_value),
        ]


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
        self.concatenated = concatenated
        return concatenated @ self.W_output

    def backward(self, dout):
        self.dW_output = self.concatenated.T @ dout
        dconcatenated = dout @ self.W_output.T
        head_grads = np.split(dconcatenated, self.num_heads, axis=1)
        dx = sum(
            head.backward(head_grad) for head, head_grad in zip(self.heads, head_grads)
        )
        return dx

    def step(self, learning_rate: float) -> None:
        self.W_output -= learning_rate * self.dW_output
        for head in self.heads:
            head.step(learning_rate)

    def parameters_and_gradients(self):
        params = [
            (self.W_output, self.dW_output),
        ]

        for head in self.heads:
            params += head.parameters_and_gradients()

        return params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
