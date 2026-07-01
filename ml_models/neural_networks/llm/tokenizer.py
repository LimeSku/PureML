import string
from typing import Any

DEFAULT_CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " \n"


class CharacterTokenizer:
    def __init__(self):
        self.char_to_id = None  # map
        self.id_to_char = None
        self.vocab_size = 0

    def fit(self, text: str = DEFAULT_CHARACTERS) -> "CharacterTokenizer":
        unique_chars = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(unique_chars)}
        self.id_to_char = {i: char for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        return self

    def encode(self, text: str) -> list[Any | int]:
        if not self.vocab_size:
            raise ValueError("Tokenizer not trained yet")
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]):
        if not self.vocab_size:
            raise ValueError("Tokenizer not trained yet")
        return "".join(self.id_to_char[token_id] for token_id in ids)


if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    tokenizer.fit()
    sentence = "Hi, im doing nonsense !"
    tokenized = tokenizer.encode(sentence)
    print(f"Initial sentence: {sentence}")
    print(f"Tokenized sentence: {tokenized}")
    print(f"Decoded sentence: {tokenizer.decode(tokenized)}")
