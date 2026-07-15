import json
from collections import Counter
from pathlib import Path


def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    merged = []
    index = 0
    while index < len(token_ids):
        if (
            index + 1 < len(token_ids)
            and (token_ids[index], token_ids[index + 1]) == pair
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1

    return merged


class BytePairTokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.token_bytes = {token_id: bytes([token_id]) for token_id in range(256)}
        self.vocab_size = 256

    def fit(self, text: str, vocab_size: int) -> "BytePairTokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        token_ids = list(text.encode("utf-8"))

        while self.vocab_size < vocab_size:
            pair_counts = Counter(zip(token_ids, token_ids[1:]))

            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )

            if pair_counts[best_pair] < 2:
                break

            new_token_id = self.vocab_size
            token_ids = merge_pair(token_ids, best_pair, new_token_id)

            left, right = best_pair
            self.merges.append(best_pair)
            self.token_bytes[new_token_id] = (
                self.token_bytes[left] + self.token_bytes[right]
            )
            self.vocab_size += 1
        return self

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        for new_token_id, pair in enumerate(self.merges, start=256):
            token_ids = merge_pair(token_ids, pair, new_token_id)
        return token_ids

    def decode(self, token_ids: list[int], errors: str = "strict") -> str:
        raw_bytes = b"".join(self.token_bytes[token_id] for token_id in token_ids)
        return raw_bytes.decode("utf-8", errors=errors)

    def save(self, path: Path) -> None:
        data = {
            "type": "byte_pair",
            "version": 1,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))

        if data["type"] != "byte_pair":
            raise ValueError(f"Unsupported tokenizer type: {data['type']}")

        tokenizer = cls()

        for serialized_pair in data["merges"]:
            left, right = serialized_pair
            pair = (left, right)
            new_token_id = tokenizer.vocab_size

            tokenizer.merges.append(pair)
            tokenizer.token_bytes[new_token_id] = (
                tokenizer.token_bytes[left] + tokenizer.token_bytes[right]
            )
            tokenizer.vocab_size += 1

        return tokenizer
