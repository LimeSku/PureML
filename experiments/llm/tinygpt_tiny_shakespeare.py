import argparse
from pathlib import Path

import numpy as np

from pureml.nn.llm.generation import generate
from pureml.nn.llm.gpt import TinyGPT
from pureml.nn.llm.losses import SequenceCrossEntropy
from pureml.nn.llm.tokenizer import CharacterTokenizer
from pureml.nn.llm.training import clip_gradients
from pureml.optimizer.adam import Adam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("datasets/tiny_shakespeare/input.txt"),
    )
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--ctx-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    rng = np.random.default_rng(42)

    text = args.path.read_text()
    train_text = text[: args.max_chars]

    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(train_text)
    if len(token_ids) <= args.ctx_length:
        raise ValueError("Training text must be longer than ctx_length")

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        ctx_length=args.ctx_length,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
    )
    loss_fn = SequenceCrossEntropy()
    optimizer = Adam(learning_rate=args.learning_rate)
    print(f"Dataset characters: {len(text)}")
    print(f"Training characters: {len(train_text)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {args.ctx_length}")
    print()

    running_loss = 0.0
    max_start = len(token_ids) - args.ctx_length - 1
    for step in range(args.steps):
        start = int(rng.integers(0, max_start))
        x = token_ids[start : start + args.ctx_length]
        y = np.array(token_ids[start + 1 : start + args.ctx_length + 1])

        logits = model(x)
        loss = loss_fn.forward(logits, y)
        dlogits = loss_fn.backward()

        model.backward(dlogits)
        clip_gradients(model.parameters_and_gradients(), max_norm=args.max_grad_norm)
        optimizer.step(model.parameters_and_gradients())
        running_loss += loss
        if step == 0 or (step + 1) % args.log_every == 0:
            avg_loss = running_loss / min(args.log_every, step + 1)
            print(f"Step {step + 1:>5}/{args.steps} - loss: {avg_loss:.4f}")
            running_loss = 0.0

    prompt_ids = tokenizer.encode(args.prompt)
    generated_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=10,
    )
    print()
    print(f"Prompt: {args.prompt!r}")
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
