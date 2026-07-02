import numpy as np

from pureml.neural_networks.llm.gpt import TinyGPT


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def generate(
    model: TinyGPT,
    token_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        context = generated[-model.ctx_length :]
        logits = model(context)

        next_token_logits = logits[-1]
        probs = softmax(next_token_logits)

        next_token_id = int(np.random.choice(len(probs), p=probs))
        generated.append(next_token_id)

    return generated
