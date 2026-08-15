import argparse
from pathlib import Path
from time import perf_counter

import torch

from pureml.llm.tokenization import CharacterTokenizer
from pureml.llm.torchgpt.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from pureml.llm.torchgpt.generation import generate
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
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.save_every <= 0:
        parser.error("--save-every must be positive")

    return args


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = select_device()

    if args.dataset == "toy":
        text = "hello world " * 100
        model_profile = "toy"
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
        if device.type == "cuda":
            model_profile = "cuda-large"
            ctx_length = 512
            embedding_dim = 512
            num_heads = 8
            num_layers = 8
            hidden_dim = 2048
        else:
            model_profile = "standard"
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

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.resume is not None:
        checkpoint_dir = args.resume.parent

    if args.resume is None:
        tokenizer = CharacterTokenizer().fit(text)
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
        start_step = 1
        best_validation_loss = float("inf")
    else:
        loaded_checkpoint = load_training_checkpoint(
            args.resume,
            device=device,
        )
        model = loaded_checkpoint.model
        optimizer = loaded_checkpoint.optimizer
        tokenizer = loaded_checkpoint.tokenizer
        ctx_length = model.ctx_length
        model_profile = "checkpoint"
        start_step = loaded_checkpoint.step + 1
        best_validation_loss = loaded_checkpoint.best_validation_loss

    try:
        encoded_text = tokenizer.encode(text)
    except KeyError as error:
        missing_character = error.args[0]
        raise ValueError(
            "dataset contains a character absent from the checkpoint tokenizer: "
            f"{missing_character!r}"
        ) from error

    token_ids = torch.tensor(
        encoded_text,
        dtype=torch.long,
        device=device,
    )

    split_index = int(len(token_ids) * 0.9)
    train_token_ids = token_ids[:split_index]
    validation_token_ids = token_ids[split_index:]

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model profile: {model_profile}")
    print(f"Dataset characters: {len(text):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(
        f"Attention heads: {model.num_heads} "
        f"(head dimension: {model.embedding_dim // model.num_heads})"
    )
    print(f"Transformer layers: {model.num_layers}")
    print(f"Feed-forward dimension: {model.hidden_dim}")
    print(f"Parameters: {parameter_count:,}")
    if args.resume is not None:
        print(f"Resumed from: {args.resume} (step {start_step - 1})")
    if checkpoint_dir is not None:
        print(f"Checkpoint directory: {checkpoint_dir}")

    synchronize_device(device)
    interval_started_at = perf_counter()
    last_logged_step = start_step - 1

    for step in range(start_step, steps + 1):
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

        logged_this_step = step == 1 or step % args.log_every == 0
        if logged_this_step:
            synchronize_device(device)
            training_elapsed = perf_counter() - interval_started_at
            interval_steps = step - last_logged_step
            seconds_per_step = training_elapsed / interval_steps

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
                f"- validation loss: {validation_loss:.4f} "
                f"- elapsed: {training_elapsed:.2f}s "
                f"- {seconds_per_step:.3f}s/step"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                if checkpoint_dir is not None:
                    best_checkpoint_path = checkpoint_dir / "best.pt"
                    save_training_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        step=step,
                        best_validation_loss=best_validation_loss,
                    )
                    print(f"Saved best checkpoint: {best_checkpoint_path}")

        should_save_last = checkpoint_dir is not None and (
            step % args.save_every == 0 or step == steps
        )
        if should_save_last:
            last_checkpoint_path = checkpoint_dir / "last.pt"
            save_training_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                step=step,
                best_validation_loss=best_validation_loss,
            )
            print(f"Saved latest checkpoint: {last_checkpoint_path}")

        if logged_this_step:
            synchronize_device(device)
            interval_started_at = perf_counter()
            last_logged_step = step

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
