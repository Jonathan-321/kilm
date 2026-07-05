import json

import torch

from kilm.analysis import summarize_tokenizer
from kilm.bpe_tokenizer import BpeTokenizer
from kilm.char_tokenizer import CharTokenizer
from kilm.corpus import load_corpus_text, load_manifest
from kilm.configs import build_model_config
from kilm.preprocessing import prepare_lines, split_lines
from kilm.reporting import render_comparison_report, render_run_report
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from kilm.tokenizers import tokenizer_from_dict
from kilm.training import learning_rate_for_step, select_device


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


def test_prepare_lines_normalizes_filters_and_dedupes():
    prepared = prepare_lines(
        "  Muraho   neza  \n\nNi\nMuraho neza\nAmakuru?\n",
        min_line_chars=3,
    )

    assert prepared.lines == ["Muraho neza", "Amakuru?"]
    assert prepared.stats["removed_blank_lines"] == 1
    assert prepared.stats["removed_short_lines"] == 1
    assert prepared.stats["removed_duplicate_lines"] == 1


def test_split_lines_is_reproducible():
    lines = [f"line {idx}" for idx in range(10)]

    train_a, val_a = split_lines(lines, val_fraction=0.2, seed=7)
    train_b, val_b = split_lines(lines, val_fraction=0.2, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_model_config_preset_allows_overrides(tmp_path):
    presets_path = tmp_path / "configs.json"
    presets_path.write_text(
        json.dumps(
            {
                "tiny": {
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 16,
                    "dropout": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    config = build_model_config(
        presets_path=presets_path,
        preset_name="tiny",
        vocab_size=10,
        overrides={"block_size": 12, "n_layer": None},
    )

    assert config.vocab_size == 10
    assert config.block_size == 12
    assert config.n_layer == 1


def test_cosine_learning_rate_decays_after_warmup():
    warmup_lr = learning_rate_for_step(
        step=1,
        max_steps=10,
        base_lr=1.0,
        min_lr=0.1,
        warmup_steps=2,
        schedule="cosine",
    )
    final_lr = learning_rate_for_step(
        step=10,
        max_steps=10,
        base_lr=1.0,
        min_lr=0.1,
        warmup_steps=2,
        schedule="cosine",
    )

    assert warmup_lr == 0.5
    assert round(final_lr, 4) == 0.1


def test_select_device_accepts_cpu():
    assert select_device("cpu").type == "cpu"


def test_generate_top_k_preserves_training_mode():
    torch.manual_seed(0)
    config = TinyTransformerConfig(
        vocab_size=8,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    model = TinyTransformerLM(config)
    model.train()

    generated = model.generate(
        torch.tensor([[0, 1]], dtype=torch.long),
        max_new_tokens=2,
        top_k=3,
    )

    assert generated.shape == (1, 4)
    assert model.training


def test_run_report_renders_key_fields():
    summary = {
        "corpus": {
            "id": "toy",
            "status": "toy",
            "path": "data/toy_corpus.txt",
            "description": "Toy corpus",
        },
        "validation_corpus": {
            "id": "toy-val",
            "status": "toy",
            "path": "data/processed/toy/val.txt",
            "description": "Toy validation split",
        },
        "tokenizer": {"type": "bpe", "vocab_size": 64, "num_merges": 24},
        "config": {"n_layer": 1, "n_head": 1, "n_embd": 16, "block_size": 8},
        "num_tokens": 20,
        "tokens_per_character": 0.75,
        "train_tokens": 16,
        "val_tokens": 8,
        "final_val_loss": 1.2,
        "final_val_perplexity": 3.32,
        "initial_val_loss": 4.2,
        "initial_val_perplexity": 66.0,
        "max_steps": 4,
        "eval_iters": 2,
        "lr_schedule": "cosine",
        "warmup_steps": 1,
        "grad_clip": 1.0,
        "checkpoint_interval": 2,
        "sample_interval": 2,
        "sample_temperature": 0.8,
        "sample_top_k": 40,
        "losses": [
            {
                "step": 2,
                "train_loss": 1.4,
                "val_loss": 1.3,
                "learning_rate": 0.01,
                "grad_norm": 0.5,
            }
        ],
        "elapsed_seconds": 0.5,
        "prompt": "Muraho",
        "sample": "Muraho neza",
        "sample_snapshots": [
            {
                "step": 2,
                "path": "sample_step_000002.txt",
                "sample": "Muraho neza",
            }
        ],
        "interpretation": "Toy only.",
        "checkpoint": "checkpoint.pt",
        "resumed_from": "previous.pt",
    }

    report = render_run_report(summary)

    assert "Final validation loss" in report
    assert "Validation Corpus" in report
    assert "Learning-rate schedule" in report
    assert "Loss Trace" in report
    assert "Sample Snapshots" in report
    assert "Muraho neza" in report
    assert "checkpoint.pt" in report
    assert "previous.pt" in report


def test_comparison_report_renders_rows():
    summary = {
        "corpus": {"id": "toy", "status": "toy"},
        "tokenizer": {"type": "char", "vocab_size": 40},
        "num_tokens": 558,
        "final_val_loss": 1.7,
        "final_val_perplexity": 5.5,
        "checkpoint": None,
    }

    report = render_comparison_report([("char_smoke", summary)])

    assert "char_smoke" in report
    assert "| char |" in report


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
