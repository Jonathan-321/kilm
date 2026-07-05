"""Evaluate tokenizer splits on morphology-focused Kinyarwanda examples."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.corpus import load_corpus_text
from kilm.tokenizers import train_tokenizer


DEFAULT_MANIFEST = ROOT / "data" / "corpora.json"
DEFAULT_EXAMPLES = ROOT / "data" / "tokenizer_eval_examples.tsv"
DEFAULT_OUT_DIR = ROOT / "experiments" / "analysis" / "morphology_tokenizers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-id", default="toy")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus, text = load_corpus_text(
        manifest_path=args.manifest,
        corpus_id=args.corpus_id,
    )
    tokenizers = {
        "char": train_tokenizer(
            "char",
            text,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_min_frequency=args.bpe_min_frequency,
        ),
        "bpe": train_tokenizer(
            "bpe",
            text,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_min_frequency=args.bpe_min_frequency,
        ),
    }
    examples = load_examples(args.examples)
    rows = []
    for example in examples:
        row = dict(example)
        row["splits"] = {}
        for name, tokenizer in tokenizers.items():
            try:
                token_ids = tokenizer.encode(example["text"])
                vocab = tokenizer.to_dict()["vocab"]
                tokens = [vocab[token_id] for token_id in token_ids]
                row["splits"][name] = {
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "token_count": len(tokens),
                }
            except ValueError as error:
                row["splits"][name] = {"error": str(error)}
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": corpus.to_dict(),
        "bpe_vocab_size": args.bpe_vocab_size,
        "bpe_min_frequency": args.bpe_min_frequency,
        "examples": rows,
    }
    json_path = args.out_dir / "morphology_tokenizer_eval.json"
    md_path = args.out_dir / "morphology_tokenizer_eval.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_report(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def load_examples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_report(payload: dict[str, object]) -> str:
    corpus = payload["corpus"]
    lines = [
        "# Morphology Tokenizer Evaluation",
        "",
        f"- Corpus: `{corpus['id']}`",
        f"- Status: `{corpus['status']}`",
        f"- BPE vocab size: `{payload['bpe_vocab_size']}`",
        "",
        "| id | text | focus | char tokens | bpe tokens | status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for example in payload["examples"]:
        char_split = example["splits"]["char"]
        bpe_split = example["splits"]["bpe"]
        lines.append(
            "| {id} | `{text}` | {focus} | {char_count} | {bpe_count} | "
            "{status} |".format(
                id=example["id"],
                text=example["text"],
                focus=example["focus"],
                char_count=char_split.get("token_count", "ERR"),
                bpe_count=bpe_split.get("token_count", "ERR"),
                status=example["status"],
            )
        )
    lines.extend(["", "## BPE Splits", ""])
    for example in payload["examples"]:
        split = example["splits"]["bpe"]
        lines.append(f"### {example['id']}")
        lines.append("")
        lines.append(f"Text: `{example['text']}`")
        if "error" in split:
            lines.append(f"Error: `{split['error']}`")
        else:
            tokens = " | ".join(str(token).replace("\n", "\\n") for token in split["tokens"])
            lines.append(f"Tokens: `{tokens}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    sys.exit(main())
