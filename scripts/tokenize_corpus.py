"""Tokenize the full Kinyarwanda corpus into Arrow datasets."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "kinyarwanda_full_corpus.txt"
DEFAULT_TOKENIZER = ROOT / "tokenizer"
DEFAULT_OUT = ROOT / "data" / "tokenized" / "kinyarwanda_spm_1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(args.corpus)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9

    dataset = load_dataset("text", data_files={"full": str(args.corpus)})["full"]
    split = dataset.train_test_split(
        test_size=args.validation_fraction,
        seed=args.seed,
        shuffle=True,
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )

    tokenized = split.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
            concatenated.append(tokenizer.eos_token_id)
        total_length = len(concatenated)
        total_length = (total_length // args.block_size) * args.block_size
        chunks = [
            concatenated[idx : idx + args.block_size]
            for idx in range(0, total_length, args.block_size)
        ]
        return {"input_ids": chunks}

    grouped = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        desc=f"Grouping into {args.block_size}-token blocks",
    )
    grouped = grouped.map(
        lambda batch: {"labels": batch["input_ids"]},
        batched=True,
        desc="Adding labels",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grouped.save_to_disk(str(args.out_dir))

    train_tokens = len(grouped["train"]) * args.block_size
    val_tokens = len(grouped["test"]) * args.block_size
    report = {
        "corpus": str(args.corpus),
        "tokenizer": str(args.tokenizer),
        "out_dir": str(args.out_dir),
        "block_size": args.block_size,
        "validation_fraction": args.validation_fraction,
        "train_blocks": len(grouped["train"]),
        "validation_blocks": len(grouped["test"]),
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
    }
    (args.out_dir / "tokenization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"out_dir={args.out_dir}")
    print(f"train_tokens={train_tokens}")
    print(f"validation_tokens={val_tokens}")
    print(f"total_tokens={train_tokens + val_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
