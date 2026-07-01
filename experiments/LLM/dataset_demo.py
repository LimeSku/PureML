from ml_models.LLM.dataset import LLMDataset
from ml_models.LLM.tokenizer import CharacterTokenizer


def main() -> None:
    text = "hello world"
    ctx_length = 4

    tokenizer = CharacterTokenizer().fit(text)
    dataset = LLMDataset.from_text(
        text=text,
        tokenizer=tokenizer,
        ctx_length=ctx_length,
    )

    print(f"Text: {text!r}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print(f"Training examples: {len(dataset)}")
    print()

    for index in range(len(dataset)):
        x, y = dataset[index]
        print(f"Example {index}")
        print(f"x: {x} -> {tokenizer.decode(x)!r}")
        print(f"y: {y} -> {tokenizer.decode(y)!r}")
        print()


if __name__ == "__main__":
    main()
