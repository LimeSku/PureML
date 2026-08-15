import torch
from torch import nn
from torch.nn import functional as F

from pureml.llm.torchgpt.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, init_std: float = 0.02):
        super().__init__()
        # replace bias term by bias in tensor
        self.W1 = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, embedding_dim, bias=True)

        nn.init.normal_(self.W1.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.W1.bias)
        nn.init.normal_(self.W2.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.W2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W1(x)
        x = F.gelu(x)
        return self.W2(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, init_std=init_std
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, init_std=init_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x
