import torch

from kilm.bpe_tokenizer import BpeTokenizer
from kilm.char_tokenizer import CharTokenizer
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM


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
