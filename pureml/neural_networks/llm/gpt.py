import numpy as np

from pureml.neural_networks.llm.embeddings import llmEmbeddingLayer
from pureml.neural_networks.llm.transformer import LayerNorm, TransformerBlock


def clip_gradients(parameters_and_gradients, max_norm: float) -> None:
    total_norm_squared = 0.0

    for _, grad in parameters_and_gradients:
        total_norm_squared += np.sum(grad**2)

    total_norm = np.sqrt(total_norm_squared)

    if total_norm <= max_norm:
        return

    scale = max_norm / (total_norm + 1e-12)

    for _, grad in parameters_and_gradients:
        grad *= scale


class TinyGPT:
    def __init__(
        self,
        vocab_size: int,
        ctx_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.ctx_length = ctx_length
        self.embedding_dim = embedding_dim

        self.embedding_layer = llmEmbeddingLayer(
            vocab_size=vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
        )
        self.blocks = [
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                init_std=init_std,
            )
            for _ in range(num_layers)
        ]

        self.final_layer_norm = LayerNorm(embedding_dim)
        self.W_output = np.random.normal(
            0.0,
            scale=init_std,
            size=(embedding_dim, vocab_size),
        )
        self.b_output = np.zeros(vocab_size)

    def forward(self, token_ids: list[int]) -> np.ndarray:
        x = self.embedding_layer(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        self.last_hidden_state = x
        logits = x @ self.W_output + self.b_output
        return logits

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        self.dW_output = self.last_hidden_state.T @ dlogits
        self.db_output = np.sum(dlogits, axis=0)
        dx = dlogits @ self.W_output.T
        dx = self.final_layer_norm.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding_layer.backward(dx)
        return dx

    def step(self, learning_rate: float) -> None:
        self.W_output -= learning_rate * self.dW_output
        self.b_output -= learning_rate * self.db_output
        self.final_layer_norm.step(learning_rate)
        for block in self.blocks:
            block.step(learning_rate)

        self.embedding_layer.step(learning_rate)

    def parameters_and_gradients(self):
        params = [
            (self.W_output, self.dW_output),
            (self.b_output, self.db_output),
        ]

        params += self.final_layer_norm.parameters_and_gradients()

        for block in self.blocks:
            params += block.parameters_and_gradients()

        params += self.embedding_layer.parameters_and_gradients()

        return params

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)
