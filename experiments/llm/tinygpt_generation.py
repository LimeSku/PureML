from pureml.neural_networks.llm.generation import generate
from pureml.neural_networks.llm.gpt import TinyGPT
from pureml.neural_networks.llm.tokenizer import (
    DEFAULT_CHARACTERS,
    CharacterTokenizer,
)


def main() -> None:
    ctx_length = 4
    prompt = "hell"
    max_new_tokens = 20

    tokenizer = CharacterTokenizer().fit(DEFAULT_CHARACTERS)
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        ctx_length=ctx_length,
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        hidden_dim=32,
    )

    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Context length: {ctx_length}")
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {tokenizer.decode(generated_ids)!r}")


if __name__ == "__main__":
    main()
