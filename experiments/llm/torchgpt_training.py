import argparse
from pathlib import Path
from time import perf_counter

import torch

from pureml.llm.tokenization import (
    BytePairTokenizer,
    CharacterTokenizer,
    TextTokenizer,
)
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

SHAKESPEARE_PATH = Path("datasets/tiny_shakespeare/input.txt")
TINY_STORIES_TRAIN_PATH = Path("datasets/tiny_stories/train.txt")
TINY_STORIES_VALIDATION_PATH = Path("datasets/tiny_stories/validation.txt")
TINY_STORIES_TRAIN_CHARACTER_LIMIT = 25_000_000
TINY_STORIES_VALIDATION_CHARACTER_LIMIT = 2_000_000
DEFAULT_BPE_VOCAB_SIZE = 1_024
DEFAULT_TOKENIZER_TRAINING_CHARACTER_LIMIT = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["toy", "shakespeare", "tinystories"],
        nargs="?",
        default="shakespeare",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="training text path (defaults depend on the selected dataset)",
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        help="validation text path for TinyStories",
    )
    parser.add_argument(
        "--max-train-characters",
        type=int,
        help="TinyStories train character limit; use 0 to load the entire file",
    )
    parser.add_argument(
        "--max-validation-characters",
        type=int,
        help="TinyStories validation character limit; use 0 to load the entire file",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["character", "bpe"],
        help="defaults to bpe for TinyStories and character for other datasets",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help=f"BPE vocabulary size (default: {DEFAULT_BPE_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--tokenizer-training-characters",
        type=int,
        help=(
            "characters used to fit BPE; use 0 for the entire training text "
            f"(default: {DEFAULT_TOKENIZER_TRAINING_CHARACTER_LIMIT})"
        ),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.vocab_size is not None and args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")
    if (
        args.tokenizer_training_characters is not None
        and args.tokenizer_training_characters < 0
    ):
        parser.error("--tokenizer-training-characters must be non-negative")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.max_train_characters is not None and args.max_train_characters < 0:
        parser.error("--max-train-characters must be non-negative")
    if (
        args.max_validation_characters is not None
        and args.max_validation_characters < 0
    ):
        parser.error("--max-validation-characters must be non-negative")
    if args.dataset != "tinystories" and args.validation_path is not None:
        parser.error("--validation-path is only supported with the tinystories dataset")
    if args.dataset != "tinystories" and (
        args.max_train_characters is not None
        or args.max_validation_characters is not None
    ):
        parser.error("character limits are only supported with the tinystories dataset")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be between 0.0 (inclusive) and 1.0 (exclusive)")
    if args.resume is not None and args.dropout is not None:
        parser.error(
            "--dropout cannot be used with --resume; it comes from the checkpoint"
        )
    if args.resume is not None and any(
        value is not None
        for value in (
            args.tokenizer,
            args.vocab_size,
            args.tokenizer_training_characters,
        )
    ):
        parser.error(
            "tokenizer options cannot be used with --resume; "
            "they come from the checkpoint"
        )

    tokenizer_type = args.tokenizer
    if tokenizer_type is None:
        tokenizer_type = "bpe" if args.dataset == "tinystories" else "character"
    if tokenizer_type == "character" and (
        args.vocab_size is not None or args.tokenizer_training_characters is not None
    ):
        parser.error("BPE options require --tokenizer bpe")

    return args


def read_text(path: Path, max_characters: int | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return file.read(max_characters)


def character_limit(value: int | None, default: int) -> int | None:
    if value == 0:
        return None
    return value if value is not None else default


def encode_text(
    tokenizer: TextTokenizer,
    text: str,
    device: torch.device,
    split: str,
) -> torch.Tensor:
    try:
        encoded_text = tokenizer.encode(text)
    except KeyError as error:
        missing_character = error.args[0]
        raise ValueError(
            f"{split} text contains a character absent from the tokenizer "
            f"vocabulary: {missing_character!r}"
        ) from error

    return torch.tensor(
        encoded_text,
        dtype=torch.long,
        device=device,
    )


def fit_tokenizer(
    text: str,
    tokenizer_type: str,
    vocab_size: int,
    training_character_limit: int | None,
) -> TextTokenizer:
    if tokenizer_type == "character":
        return CharacterTokenizer().fit(text)

    training_text = (
        text if training_character_limit is None else text[:training_character_limit]
    )
    return BytePairTokenizer().fit(training_text, vocab_size=vocab_size)


def tokenizer_name(tokenizer: TextTokenizer) -> str:
    if isinstance(tokenizer, CharacterTokenizer):
        return "character"
    if isinstance(tokenizer, BytePairTokenizer):
        return "byte_pair"
    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__name__}")


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
        split_index = int(len(text) * 0.9)
        training_text = text[:split_index]
        validation_text = text[split_index:]
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
        if args.dataset == "shakespeare":
            path = args.path if args.path is not None else SHAKESPEARE_PATH
            text = read_text(path)
            split_index = int(len(text) * 0.9)
            training_text = text[:split_index]
            validation_text = text[split_index:]
            default_steps = 2_000
            prompt = "ROMEO:"
        else:
            training_path = (
                args.path if args.path is not None else TINY_STORIES_TRAIN_PATH
            )
            validation_path = (
                args.validation_path
                if args.validation_path is not None
                else TINY_STORIES_VALIDATION_PATH
            )
            training_text = read_text(
                training_path,
                character_limit(
                    args.max_train_characters,
                    TINY_STORIES_TRAIN_CHARACTER_LIMIT,
                ),
            )
            validation_text = read_text(
                validation_path,
                character_limit(
                    args.max_validation_characters,
                    TINY_STORIES_VALIDATION_CHARACTER_LIMIT,
                ),
            )
            default_steps = 10_000
            prompt = "Once upon a time"

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
        steps = args.steps if args.steps is not None else default_steps
        max_new_tokens = 300

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.resume is not None:
        checkpoint_dir = args.resume.parent

    if args.resume is None:
        dropout = args.dropout if args.dropout is not None else 0.1
        selected_tokenizer = args.tokenizer
        if selected_tokenizer is None:
            selected_tokenizer = "bpe" if args.dataset == "tinystories" else "character"
        vocab_size = (
            args.vocab_size if args.vocab_size is not None else DEFAULT_BPE_VOCAB_SIZE
        )
        tokenizer_training_limit = character_limit(
            args.tokenizer_training_characters,
            DEFAULT_TOKENIZER_TRAINING_CHARACTER_LIMIT,
        )
        tokenizer_started_at = perf_counter()
        tokenizer = fit_tokenizer(
            text=training_text,
            tokenizer_type=selected_tokenizer,
            vocab_size=vocab_size,
            training_character_limit=tokenizer_training_limit,
        )
        tokenizer_training_elapsed = perf_counter() - tokenizer_started_at
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=steps,
            eta_min=3e-5,
        )
        start_step = 1
        best_validation_loss = float("inf")
    else:
        tokenizer_training_elapsed = None
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

        if loaded_checkpoint.scheduler is None:
            remaining_steps = max(1, steps - loaded_checkpoint.step)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=remaining_steps,
                eta_min=3e-5,
            )
            print(
                "Checkpoint has no scheduler state; initialized cosine decay "
                f"for the remaining {remaining_steps} step(s)."
            )
        else:
            scheduler = loaded_checkpoint.scheduler
            scheduler_target_step = loaded_checkpoint.step + max(
                scheduler.T_max - scheduler.last_epoch,
                0,
            )
            if args.steps is None:
                steps = scheduler_target_step
            elif steps != scheduler_target_step:
                raise ValueError(
                    f"--steps must be {scheduler_target_step} when resuming this "
                    "scheduler; start a new training run to use a different horizon"
                )

    train_token_ids = encode_text(
        tokenizer=tokenizer,
        text=training_text,
        device=device,
        split="training",
    )
    validation_token_ids = encode_text(
        tokenizer=tokenizer,
        text=validation_text,
        device=device,
        split="validation",
    )
    training_character_count = len(training_text)
    validation_character_count = len(validation_text)
    del training_text, validation_text

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model profile: {model_profile}")
    print(f"Training characters: {training_character_count:,}")
    print(f"Validation characters: {validation_character_count:,}")
    print(f"Tokenizer: {tokenizer_name(tokenizer)}")
    if tokenizer_training_elapsed is not None:
        print(f"Tokenizer training: {tokenizer_training_elapsed:.2f}s")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(train_token_ids):,}")
    print(f"Validation tokens: {len(validation_token_ids):,}")
    print(f"Context length: {ctx_length}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(
        f"Attention heads: {model.num_heads} "
        f"(head dimension: {model.embedding_dim // model.num_heads})"
    )
    print(f"Transformer layers: {model.num_layers}")
    print(f"Feed-forward dimension: {model.hidden_dim}")
    print(f"Dropout: {model.dropout}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Scheduler: cosine decay to {scheduler.eta_min:.2e}")
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
        scheduler.step()

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
                f"- learning rate: {scheduler.get_last_lr()[0]:.2e} "
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
                        scheduler=scheduler,
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
                scheduler=scheduler,
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
    print(tokenizer.decode(generated_ids, errors="replace"))


if __name__ == "__main__":
    main()
