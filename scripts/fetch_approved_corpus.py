"""Fetch approved Kinyarwanda text corpora into local ignored files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.preprocessing import prepare_lines


@dataclass(frozen=True)
class SourceSpec:
    id: str
    description: str
    license: str
    source_url: str
    files: tuple[str, ...]
    delimiter: str
    text_column: int
    skip_header: bool = False


SOURCES = {
    "digital-umuganda-tts-rw": SourceSpec(
        id="digital-umuganda-tts-rw",
        description="Kinyarwanda TTS sentence text from DigitalUmuganda.",
        license="cc0-1.0",
        source_url=(
            "https://huggingface.co/datasets/"
            "DigitalUmuganda/kinyarwanda-tts-dataset"
        ),
        files=(
            "https://huggingface.co/datasets/"
            "DigitalUmuganda/kinyarwanda-tts-dataset/resolve/main/"
            "4_000_sentences.csv",
        ),
        delimiter=",",
        text_column=1,
    ),
    "digital-umuganda-mt-rw": SourceSpec(
        id="digital-umuganda-mt-rw",
        description=(
            "Kinyarwanda side of DigitalUmuganda Kinyarwanda-English "
            "parallel text."
        ),
        license="cc-by-4.0",
        source_url=(
            "https://huggingface.co/datasets/DigitalUmuganda/"
            "kinyarwanda-english-machine-translation-dataset"
        ),
        files=(
            "https://huggingface.co/datasets/DigitalUmuganda/"
            "kinyarwanda-english-machine-translation-dataset/resolve/main/"
            "kinyarwanda-english-corpus.tsv",
            "https://huggingface.co/datasets/DigitalUmuganda/"
            "kinyarwanda-english-machine-translation-dataset/resolve/main/"
            "kinyarwanda-english-corpus2.tsv",
            "https://huggingface.co/datasets/DigitalUmuganda/"
            "kinyarwanda-english-machine-translation-dataset/resolve/main/"
            "kinyarwanda-english-corpus3.tsv",
        ),
        delimiter="\t",
        text_column=0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "approved")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-line-chars", type=int, default=3)
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = SOURCES[args.source]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for url in spec.files:
        raw_path = args.raw_dir / f"{spec.id}_{Path(url).name}"
        fetch(url, raw_path)
        rows.extend(extract_text(raw_path, spec))
        if args.limit is not None and len(rows) >= args.limit:
            rows = rows[: args.limit]
            break

    prepared = prepare_lines(
        "\n".join(rows),
        min_line_chars=args.min_line_chars,
        dedupe=not args.keep_duplicates,
    )
    output_stem = spec.id.replace("-", "_")
    text_path = args.out_dir / f"{output_stem}.txt"
    card_path = args.out_dir / f"{output_stem}_source_card.md"
    text_path.write_text(prepared.text, encoding="utf-8")
    card_path.write_text(
        render_source_card(spec, prepared.stats, text_path),
        encoding="utf-8",
    )

    print(f"source={spec.id}")
    print(f"license={spec.license}")
    print(f"prepared_lines={prepared.stats['prepared_lines']}")
    print(f"removed_duplicate_lines={prepared.stats['removed_duplicate_lines']}")
    print(f"wrote {text_path}")
    print(f"wrote {card_path}")
    return 0


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    subprocess.run(
        ["curl", "-L", "--fail", "--max-time", "120", "-o", str(path), url],
        check=True,
    )


def extract_text(path: Path, spec: SourceSpec) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(text.splitlines(), delimiter=spec.delimiter)
    lines: list[str] = []
    for row_idx, row in enumerate(reader):
        if spec.skip_header and row_idx == 0:
            continue
        if len(row) <= spec.text_column:
            continue
        value = row[spec.text_column].strip()
        if value:
            lines.append(value)
    return lines


def render_source_card(
    spec: SourceSpec,
    stats: dict[str, int | bool],
    text_path: Path,
) -> str:
    lines = [
        "# Source Card",
        "",
        f"- ID: `{spec.id}`",
        f"- Description: {spec.description}",
        f"- Source: {spec.source_url}",
        f"- License: `{spec.license}`",
        f"- Local text path: `{text_path}`",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- {url}" for url in spec.files)
    lines.extend(
        [
            "",
            "## Preparation Stats",
            "",
            f"- Raw lines: `{stats['raw_lines']}`",
            f"- Prepared lines: `{stats['prepared_lines']}`",
            f"- Removed blank lines: `{stats['removed_blank_lines']}`",
            f"- Removed short lines: `{stats['removed_short_lines']}`",
            f"- Removed duplicate lines: `{stats['removed_duplicate_lines']}`",
            "",
            "## Use Notes",
            "",
            "Generated text files are local artifacts and should stay out of Git.",
            "Preserve source attribution in reports and presentations.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    sys.exit(main())
