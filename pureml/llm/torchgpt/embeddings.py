import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, ctx_length: int, embedding_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.context_length = ctx_length
        self.dropout = nn.Dropout(dropout)

        self.token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding_layer = nn.Embedding(
            ctx_length,
            embedding_dim,
        )
        # init the weights as before
        nn.init.normal_(self.token_embedding_layer.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding_layer.weight, mean=0.0, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        batch_size, sequence_length = token_ids.shape
        # positions = np.broadcast_to(
        #     np.arange(sequence_length),
        #     (batch_size, sequence_length),
        # )
        positions = torch.arange(sequence_length, device=token_ids.device)
        token_embeddings = self.token_embedding_layer(token_ids)
        pos_embeddings = self.pos_embedding_layer(positions)
        # unsqueeze adds dimension of 1 at pos 0
        return self.dropout(token_embeddings + pos_embeddings.unsqueeze(0))
