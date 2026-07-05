"""A small character-seeded BPE tokenizer for the KILM sandbox."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


Pair = tuple[str, str]


@dataclass(frozen=True)
class BpeTokenizer:
    """Reversible BPE tokenizer trained from raw text characters."""

    vocab: tuple[str, ...]
    merges: tuple[Pair, ...]
    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vocab:
            raise ValueError("vocab must not be empty")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocab entries must be unique")
        if not self.chars:
            raise ValueError("chars must not be empty")
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("chars must be unique")

    @classmethod
    def train(
        cls,
        text: str,
        *,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        if not text:
            raise ValueError("cannot train tokenizer on empty text")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")

        chars = tuple(sorted(set(text)))
        if vocab_size < len(chars):
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than character vocabulary "
                f"size {len(chars)}"
            )

        symbols = list(text)
        vocab = list(chars)
        vocab_set = set(vocab)
        merges: list[Pair] = []

        while len(vocab) < vocab_size:
            pair_counts = Counter(zip(symbols, symbols[1:], strict=False))
            if not pair_counts:
                break

            best_pair, best_count = sorted(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            if best_count < min_frequency:
                break

            merged = "".join(best_pair)
            symbols = _merge_pair(symbols, best_pair, merged)
            merges.append(best_pair)
            if merged not in vocab_set:
                vocab.append(merged)
                vocab_set.add(merged)

        return cls(vocab=tuple(vocab), merges=tuple(merges), chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def stoi(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.vocab)}

    @property
    def itos(self) -> dict[int, str]:
        return {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        known_chars = set(self.chars)
        unknown_chars = sorted(set(text) - known_chars)
        if unknown_chars:
            raise ValueError(f"unknown character: {unknown_chars[0]!r}")

        symbols = list(text)
        for left, right in self.merges:
            symbols = _merge_pair(symbols, (left, right), left + right)

        stoi = self.stoi
        try:
            return [stoi[symbol] for symbol in symbols]
        except KeyError as error:
            raise ValueError(f"unknown token: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        itos = self.itos
        try:
            return "".join(itos[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]!r}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "bpe",
            "vocab": list(self.vocab),
            "chars": list(self.chars),
            "merges": [list(pair) for pair in self.merges],
        }


def _merge_pair(symbols: list[str], pair: Pair, merged: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(symbols):
        if idx < len(symbols) - 1 and (symbols[idx], symbols[idx + 1]) == pair:
            out.append(merged)
            idx += 2
        else:
            out.append(symbols[idx])
            idx += 1
    return out
