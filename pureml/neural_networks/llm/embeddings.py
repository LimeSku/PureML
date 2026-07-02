import numpy as np


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 0.02):
        """
        vocab_size: number of tokens in the vocabulary
        embedding_dim: size of each token vector
        init_std=0.02 following GPT2 embeddings init
        """
        self.vocab_size = num_embeddings
        self.embedding_dim = embedding_dim

        self.weights = np.random.normal(
            0.0,
            scale=init_std,
            size=(num_embeddings, embedding_dim),
        )

    def forward(self, token_ids: list[int]) -> np.ndarray:
        return self.weights[token_ids]

    def __call__(self, token_ids: list[int]):
        return self.forward(token_ids)


class llmEmbeddingLayer:
    def __init__(self, vocab_size: int, ctx_length: int, embedding_dim: int):
        self.token_embedding_layer = Embedding(vocab_size, embedding_dim)
        self.pos_embedding_layer = Embedding(ctx_length, embedding_dim, init_std=0.01)

    def forward(self, token_ids: list[int]) -> np.ndarray:
        positions = list(range(len(token_ids)))
        token_embeddings = self.token_embedding_layer(token_ids)
        pos_embeddings = self.pos_embedding_layer(positions)
        return token_embeddings + pos_embeddings

    def __call__(self, token_ids: list[int]) -> np.ndarray:
        return self.forward(token_ids)
