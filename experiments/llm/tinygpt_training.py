import numpy as np

from pureml.neural_networks.llm.dataset import llmDataset
from pureml.neural_networks.llm.generation import generate
from pureml.neural_networks.llm.gpt import TinyGPT
from pureml.neural_networks.llm.losses import SequenceCrossEntropy
from pureml.neural_networks.llm.tokenizer import CharacterTokenizer


def main() -> None:
    text = "hello world hello world hello world"
    ctx_length = 4
    learning_rate = 0.1
    epochs = 200

    tokenizer = CharacterTokenizer().fit(text)
    dataset = llmDataset.from_text(
        text=text,
        tokenizer=tokenizer,
        ctx_length=ctx_length,
    )
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        ctx_length=ctx_length,
        embedding_dim=8,
        num_heads=2,
        num_layers=2,
        hidden_dim=32,
    )
    loss_fn = SequenceCrossEntropy()

    for epoch in range(epochs):
        total_loss = 0.0
        for index in range(len(dataset)):
            x, y = dataset[index]
            targets = np.array(y)

            logits = model(x)
            loss = loss_fn.forward(logits, targets)
            dlogits = loss_fn.backward()

            model.backward(dlogits)
            model.step(learning_rate=learning_rate)

            total_loss += loss

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:>3}/{epochs} - loss: {total_loss / len(dataset):.4f}"
            )

    prompt = "hell"
    generated_ids = generate(
        model=model,
        token_ids=tokenizer.encode(prompt),
        max_new_tokens=24,
        temperature=0.8,
        top_k=10,
    )
    print()
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {tokenizer.decode(generated_ids)!r}")


if __name__ == "__main__":
    main()
