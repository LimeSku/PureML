import numpy as np
import pytest

from pureml.llm.tinygpt.embeddings import Embedding, llmEmbeddingLayer


def test_embedding_accepts_batched_token_ids() -> None:
    embedding = Embedding(num_embeddings=4, embedding_dim=2)
    embedding.weights = np.arange(8).reshape(4, 2)
    token_ids = np.array([[0, 2], [3, 1]])

    output = embedding(token_ids)

    np.testing.assert_array_equal(output, embedding.weights[token_ids])
    assert output.shape == (2, 2, 2)


def test_embedding_accumulates_repeated_token_gradients() -> None:
    embedding = Embedding(num_embeddings=3, embedding_dim=2)
    token_ids = np.array([[0, 1], [0, 2]])
    output = embedding(token_ids)

    embedding.backward(np.ones_like(output))

    expected = np.array(
        [
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(embedding.dweights, expected)


def test_llm_embedding_layer_adds_positions_to_each_batch_item() -> None:
    layer = llmEmbeddingLayer(vocab_size=4, ctx_length=3, embedding_dim=2)
    layer.token_embedding_layer.weights.fill(0.0)
    layer.pos_embedding_layer.weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    token_ids = np.array([[0, 1, 2], [2, 1, 0]])

    output = layer(token_ids)

    expected = np.broadcast_to(
        layer.pos_embedding_layer.weights,
        (2, 3, 2),
    )
    np.testing.assert_array_equal(output, expected)


def test_llm_embedding_layer_accumulates_position_gradients_across_batch() -> None:
    layer = llmEmbeddingLayer(vocab_size=4, ctx_length=3, embedding_dim=2)
    token_ids = np.array([[0, 1, 2], [2, 1, 0]])
    output = layer(token_ids)

    layer.backward(np.ones_like(output))

    expected = np.full((3, 2), 2.0)
    np.testing.assert_array_equal(layer.pos_embedding_layer.dweights, expected)


def test_llm_embedding_layer_rejects_unbatched_token_ids() -> None:
    layer = llmEmbeddingLayer(vocab_size=4, ctx_length=3, embedding_dim=2)

    with pytest.raises(
        ValueError,
        match="token_ids must have shape",
    ):
        layer(np.array([0, 1, 2]))
