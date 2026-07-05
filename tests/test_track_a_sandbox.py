import json

import torch

from kilm.analysis import summarize_tokenizer
from kilm.bpe_tokenizer import BpeTokenizer
from kilm.char_tokenizer import CharTokenizer
from kilm.corpus import load_corpus_text, load_manifest
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from kilm.tokenizers import tokenizer_from_dict


def test_char_tokenizer_round_trips_text():
    tokenizer = CharTokenizer.train("Muraho\nAmakuru?")
    text = "Muraho"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_rejects_unknown_characters():
    tokenizer = CharTokenizer.train("abc")

    try:
        tokenizer.encode("abcd")
    except ValueError as error:
        assert "unknown character" in str(error)
    else:
        raise AssertionError("expected unknown character failure")


def test_char_tokenizer_serializes_round_trip():
    tokenizer = CharTokenizer.train("Muraho")
    restored = tokenizer_from_dict(tokenizer.to_dict())

    assert restored.decode(restored.encode("Muraho")) == "Muraho"


def test_bpe_tokenizer_round_trips_text():
    text = "Muraho Muraho\nAmakuru?"
    tokenizer = BpeTokenizer.train(text, vocab_size=24)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_learns_repeated_merges():
    tokenizer = BpeTokenizer.train("banana banana banana", vocab_size=12)

    assert tokenizer.vocab_size > len(tokenizer.chars)
    assert tokenizer.encode("banana banana")
    assert tokenizer.decode(tokenizer.encode("banana banana")) == "banana banana"


def test_bpe_tokenizer_rejects_unknown_characters():
    tokenizer = BpeTokenizer.train("abc abc", vocab_size=6)

    try:
        tokenizer.encode("abcd")
    except ValueError as error:
        assert "unknown character" in str(error)
    else:
        raise AssertionError("expected unknown character failure")


def test_bpe_tokenizer_serializes_round_trip():
    tokenizer = BpeTokenizer.train("banana banana banana", vocab_size=12)
    restored = tokenizer_from_dict(tokenizer.to_dict())

    assert restored.decode(restored.encode("banana banana")) == "banana banana"


def test_manifest_loads_toy_corpus(tmp_path):
    corpus_path = tmp_path / "toy.txt"
    corpus_path.write_text("Muraho\nAmakuru\n", encoding="utf-8")
    manifest_path = tmp_path / "corpora.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": [
                    {
                        "id": "toy",
                        "path": "toy.txt",
                        "status": "toy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_manifest(manifest_path)
    record, text = load_corpus_text(manifest_path=manifest_path, corpus_id="toy")

    assert records["toy"].status == "toy"
    assert record.path == corpus_path.resolve()
    assert "Muraho" in text


def test_manifest_blocks_unapproved_corpus(tmp_path):
    corpus_path = tmp_path / "blocked.txt"
    corpus_path.write_text("Muraho\n", encoding="utf-8")
    manifest_path = tmp_path / "corpora.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": [
                    {
                        "id": "blocked",
                        "path": "blocked.txt",
                        "status": "blocked",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        load_corpus_text(manifest_path=manifest_path, corpus_id="blocked")
    except ValueError as error:
        assert "allow-unapproved" in str(error)
    else:
        raise AssertionError("expected blocked corpus failure")


def test_tokenizer_summary_reports_compression():
    text = "Muraho Muraho Muraho"
    char = CharTokenizer.train(text)
    bpe = BpeTokenizer.train(text, vocab_size=16)

    char_summary = summarize_tokenizer("char", char, text)
    bpe_summary = summarize_tokenizer("bpe", bpe, text)

    assert bpe_summary["num_tokens"] < char_summary["num_tokens"]
    assert bpe_summary["tokens_per_word"] < char_summary["tokens_per_word"]


def test_tiny_transformer_forward_shapes():
    config = TinyTransformerConfig(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = TinyTransformerLM(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    y = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(x, y)

    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
