import argparse
from pathlib import Path

import numpy as np

from pureml.llm.generation import generate
from pureml.llm.losses import SequenceCrossEntropy
from pureml.llm.tinygpt.model import TinyGPT
from pureml.llm.tokenization.char_tokenizer import CharacterTokenizer
from pureml.llm.training import train_language_model_step
from pureml.optimizer.adam import Adam
from pureml.optimizer.sgd import SGD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["shakespeare", "toy"],
        nargs="?",
        default="shakespeare",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("datasets/tiny_shakespeare/input.txt"),
    )
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--ctx-length", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--load-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    rng = np.random.default_rng(42)

    if args.dataset == "toy":
        text = "hello world hello world hello world"
        train_text = text
        ctx_length = args.ctx_length if args.ctx_length is not None else 4
        steps = args.steps
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.1
        log_every = args.log_every
        prompt = args.prompt if args.prompt is not None else "hell"
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 24
        embedding_dim = 8
        num_heads = 2
        num_layers = 2
        hidden_dim = 32
        optimizer = SGD(learning_rate=learning_rate)
    else:
        text = args.path.read_text()
        train_text = text[: args.max_chars]
        ctx_length = args.ctx_length if args.ctx_length is not None else 32
        steps = args.steps if args.steps is not None else 1_000
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
        log_every = args.log_every if args.log_every is not None else 100
        prompt = args.prompt if args.prompt is not None else "ROMEO:"
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 200
        embedding_dim = 128
        num_heads = 4
        num_layers = 4
        hidden_dim = 512
        optimizer = Adam(learning_rate=learning_rate)

    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(train_text)

    if args.load_checkpoint is not None:
        model = TinyGPT.from_checkpoint(args.load_checkpoint)
        if args.ctx_length is not None and args.ctx_length != model.ctx_length:
            raise ValueError(
                "ctx_length does not match the loaded checkpoint: "
                f"expected {model.ctx_length}, got {args.ctx_length}"
            )
        if tokenizer.vocab_size != model.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size does not match the loaded checkpoint: "
                f"expected {model.vocab_size}, got {tokenizer.vocab_size}"
            )
        ctx_length = model.ctx_length
    else:
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            ctx_length=ctx_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )

    if len(token_ids) <= ctx_length:
        raise ValueError("Training text must be longer than ctx_length")
    if args.dataset == "toy":
        if args.steps is None:
            steps = 200 * (len(train_text) - ctx_length)
        if args.log_every is None:
            log_every = max(1, len(train_text) - ctx_length)
    loss_fn = SequenceCrossEntropy()

    print(f"Dataset: {args.dataset}")
    print(f"Dataset characters: {len(text)}")
    print(f"Training characters: {len(train_text)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print()

    running_loss = 0.0
    max_start = len(token_ids) - ctx_length
    for step in range(steps):
        start = int(rng.integers(0, max_start))
        x = token_ids[start : start + ctx_length]
        y = np.array(token_ids[start + 1 : start + ctx_length + 1])
        loss = train_language_model_step(
            model,
            loss_fn,
            optimizer,
            x,
            y,
            max_grad_norm=args.max_grad_norm,
        )

        running_loss += loss
        if step == 0 or (step + 1) % log_every == 0:
            avg_loss = running_loss / min(log_every, step + 1)
            print(f"Step {step + 1:>5}/{steps} - loss: {avg_loss:.4f}")
            running_loss = 0.0

    if args.save_checkpoint is not None:
        model.save_checkpoint(args.save_checkpoint)

    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print()
    print(f"Prompt: {prompt!r}")
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
