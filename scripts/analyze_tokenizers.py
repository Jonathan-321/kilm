"""Compare KILM tokenizer behavior on a corpus."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.analysis import render_tokenizer_report, summarize_tokenizer
from kilm.corpus import load_corpus_text
from kilm.tokenizers import train_tokenizer


DEFAULT_MANIFEST = ROOT / "data" / "corpora.json"
DEFAULT_OUT_DIR = ROOT / "experiments" / "analysis" / "tokenizers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-id", default="toy")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--allow-unapproved-corpus", action="store_true")
    parser.add_argument("--bpe-vocab-size", type=int, default=64)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus, text = load_corpus_text(
        manifest_path=args.manifest,
        corpus_id=args.corpus_id,
        corpus_path=args.corpus,
        allow_unapproved=args.allow_unapproved_corpus,
    )
    tokenizers = [
        (
            "char",
            train_tokenizer(
                "char",
                text,
                bpe_vocab_size=args.bpe_vocab_size,
                bpe_min_frequency=args.bpe_min_frequency,
            ),
        ),
        (
            "bpe",
            train_tokenizer(
                "bpe",
                text,
                bpe_vocab_size=args.bpe_vocab_size,
                bpe_min_frequency=args.bpe_min_frequency,
            ),
        ),
    ]
    summaries = [
        summarize_tokenizer(name, tokenizer, text, top_k=args.top_k)
        for name, tokenizer in tokenizers
    ]
    payload = {
        "corpus": corpus.to_dict(),
        "bpe_vocab_size": args.bpe_vocab_size,
        "bpe_min_frequency": args.bpe_min_frequency,
        "summaries": summaries,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "tokenizer_analysis.json"
    md_path = args.out_dir / "tokenizer_analysis.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_tokenizer_report(corpus=corpus, summaries=summaries),
        encoding="utf-8",
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
