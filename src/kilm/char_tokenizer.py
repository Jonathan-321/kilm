"""A tiny character tokenizer for the Track A sandbox."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Reversible character-level tokenizer."""

    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("chars must be unique")
        if not self.chars:
            raise ValueError("chars must not be empty")

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("cannot train tokenizer on empty text")
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: idx for idx, char in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {idx: char for idx, char in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        stoi = self.stoi
        try:
            return [stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        itos = self.itos
        try:
            return "".join(itos[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]!r}") from error
