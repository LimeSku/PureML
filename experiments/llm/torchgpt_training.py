import argparse
from pathlib import Path

import torch

from pureml.llm.tokenization import CharacterTokenizer
from pureml.llm.torchgpt.model import TinyGPT
from pureml.llm.torchgpt.training import (
    language_model_loss,
    train_language_model_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["toy", "shakespeare"],
        nargs="?",
        default="shakespeare",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("datasets/tiny_shakespeare/input.txt"),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sample_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    ctx_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(token_ids) - ctx_length
    if max_start <= 0:
        raise ValueError("token_ids must contain more tokens than ctx_length")

    starts = torch.randint(
        0,
        max_start,
        (batch_size,),
        device=token_ids.device,
    )
    offsets = torch.arange(
        ctx_length,
        device=token_ids.device,
    )
    indices = starts[:, None] + offsets[None, :]

    x_batch = token_ids[indices]
    y_batch = token_ids[indices + 1]
    return x_batch, y_batch


@torch.no_grad()
def evaluate_language_model(
    model: TinyGPT,
    token_ids: torch.Tensor,
    batch_size: int,
    ctx_length: int,
    num_batches: int,
) -> float:
    model.eval()
    losses = []

    for _ in range(num_batches):
        x_batch, y_batch = sample_batch(
            token_ids=token_ids,
            batch_size=batch_size,
            ctx_length=ctx_length,
        )
        logits = model(x_batch)
        losses.append(language_model_loss(logits, y_batch))

    return torch.stack(losses).mean().item()


@torch.no_grad()
def generate(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[int]:
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = select_device()

    if args.dataset == "toy":
        text = "hello world " * 100
        ctx_length = 16
        embedding_dim = 64
        num_heads = 4
        num_layers = 2
        hidden_dim = 256
        learning_rate = 3e-3
        weight_decay = 0.0
        steps = args.steps if args.steps is not None else 1_000
        prompt = "hell"
        max_new_tokens = 40
    else:
        text = args.path.read_text(encoding="utf-8")
        ctx_length = 256
        embedding_dim = 384
        num_heads = 6
        num_layers = 6
        hidden_dim = 1536
        learning_rate = 3e-4
        weight_decay = 0.01
        steps = args.steps if args.steps is not None else 2_000
        prompt = "ROMEO:"
        max_new_tokens = 300

    tokenizer = CharacterTokenizer().fit(text)

    token_ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    )

    split_index = int(len(token_ids) * 0.9)
    train_token_ids = token_ids[:split_index]
    validation_token_ids = token_ids[split_index:]

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        ctx_length=ctx_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset characters: {len(text):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print(f"Parameters: {parameter_count:,}")

    for step in range(1, steps + 1):
        x_batch, y_batch = sample_batch(
            token_ids=train_token_ids,
            batch_size=args.batch_size,
            ctx_length=ctx_length,
        )
        loss = train_language_model_step(
            model=model,
            optimizer=optimizer,
            x_batch=x_batch,
            y_batch=y_batch,
            max_grad_norm=1.0,
        )

        if step == 1 or step % args.log_every == 0:
            validation_loss = evaluate_language_model(
                model=model,
                token_ids=validation_token_ids,
                batch_size=args.batch_size,
                ctx_length=ctx_length,
                num_batches=args.eval_batches,
            )
            print(
                f"Step {step:>4}/{steps} "
                f"- train loss: {loss.item():.4f} "
                f"- validation loss: {validation_loss:.4f}"
            )

    generated_ids = generate(
        model=model,
        prompt_ids=tokenizer.encode(prompt),
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    print()
    print(f"Prompt: {prompt!r}")
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
