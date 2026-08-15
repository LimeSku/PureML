import torch

from pureml.llm.torchgpt.model import TinyGPT


@torch.no_grad()
def generate(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        context = generated[:, -model.ctx_length :]
        logits = model(context)
        next_token_logits = logits[:, -1] / temperature
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )
        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated[0].tolist()
