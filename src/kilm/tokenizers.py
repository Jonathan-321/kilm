"""Shared tokenizer construction and serialization helpers."""

from __future__ import annotations

from kilm.bpe_tokenizer import BpeTokenizer
from kilm.char_tokenizer import CharTokenizer


AnyTokenizer = CharTokenizer | BpeTokenizer


def train_tokenizer(
    kind: str,
    text: str,
    *,
    bpe_vocab_size: int,
    bpe_min_frequency: int,
) -> AnyTokenizer:
    if kind == "char":
        return CharTokenizer.train(text)
    if kind == "bpe":
        return BpeTokenizer.train(
            text,
            vocab_size=bpe_vocab_size,
            min_frequency=bpe_min_frequency,
        )
    raise ValueError(f"unknown tokenizer kind: {kind}")


def tokenizer_from_dict(payload: dict[str, object]) -> AnyTokenizer:
    tokenizer_type = payload.get("type")
    if tokenizer_type == "char":
        return CharTokenizer.from_dict(payload)
    if tokenizer_type == "bpe":
        return BpeTokenizer.from_dict(payload)
    raise ValueError(f"unknown tokenizer type: {tokenizer_type!r}")
