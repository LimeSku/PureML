import numpy as np

from ml_models.LLM.embeddings import LLMEmbeddingLayer
from ml_models.LLM.transformer import LayerNorm, TransformerBlock


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

        self.embedding_layer = LLMEmbeddingLayer(
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
        logits = x @ self.W_output + self.b_output
        return logits

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward(token_ids)
