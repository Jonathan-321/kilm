"""Prepare a KILM corpus into cleaned train/validation files."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.corpus import load_corpus_text
from kilm.preprocessing import prepare_lines, render_corpus_card, split_lines


DEFAULT_MANIFEST = ROOT / "data" / "corpora.json"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "toy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-id", default="toy")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--allow-unapproved-corpus", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-line-chars", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, text = load_corpus_text(
        manifest_path=args.manifest,
        corpus_id=args.corpus_id,
        corpus_path=args.corpus,
        allow_unapproved=args.allow_unapproved_corpus,
    )
    prepared = prepare_lines(
        text,
        min_line_chars=args.min_line_chars,
        dedupe=args.dedupe,
    )
    train_lines, val_lines = split_lines(
        prepared.lines,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "full.txt").write_text(prepared.text, encoding="utf-8")
    (args.out_dir / "train.txt").write_text(
        "\n".join(train_lines).strip() + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "val.txt").write_text(
        "\n".join(val_lines).strip() + "\n",
        encoding="utf-8",
    )
    stats = {
        "source": source.to_dict(),
        "stats": prepared.stats,
        "train_lines": len(train_lines),
        "val_lines": len(val_lines),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "corpus_card.md").write_text(
        render_corpus_card(
            source=source,
            prepared=prepared,
            train_lines=train_lines,
            val_lines=val_lines,
            val_fraction=args.val_fraction,
            seed=args.seed,
        ),
        encoding="utf-8",
    )
    manifest = {
        "corpora": [
            _prepared_record(source, args.out_dir, "full", "full.txt"),
            _prepared_record(source, args.out_dir, "train", "train.txt"),
            _prepared_record(source, args.out_dir, "val", "val.txt"),
        ]
    }
    (args.out_dir / "corpora.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"prepared_lines={prepared.stats['prepared_lines']}")
    print(f"train_lines={len(train_lines)}")
    print(f"val_lines={len(val_lines)}")
    print(f"wrote {args.out_dir}")
    return 0


def _prepared_record(
    source,
    out_dir: Path,
    split: str,
    filename: str,
) -> dict[str, str]:
    return {
        "id": f"{source.id}-{split}",
        "path": filename,
        "status": source.status,
        "description": f"{split} split prepared from {source.id}",
        "source": source.source,
        "license": source.license,
        "notes": (
            f"Prepared in {out_dir}. Inherits source status {source.status!r}."
        ),
    }


if __name__ == "__main__":
    sys.exit(main())
