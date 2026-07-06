"""Aggregate a large Kinyarwanda corpus from documented open sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import sys
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "kinyarwanda_full_corpus.txt"
DEFAULT_REPORT = ROOT / "data" / "kinyarwanda_full_corpus_stats.json"

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ’']+")

KINYA_MARKERS = {
    "abanyarwanda",
    "afite",
    "aho",
    "ari",
    "ariko",
    "avuga",
    "bari",
    "bose",
    "bwa",
    "bwo",
    "bya",
    "cyane",
    "cyangwa",
    "gukora",
    "hari",
    "ibi",
    "ibyo",
    "icyo",
    "igihe",
    "iki",
    "iyo",
    "kandi",
    "kigali",
    "kinyarwanda",
    "ko",
    "kuba",
    "kubera",
    "kuko",
    "kuri",
    "mu",
    "muri",
    "na",
    "ngo",
    "nta",
    "ntabwo",
    "nubwo",
    "rwanda",
    "uko",
    "uyu",
    "uwo",
    "yagize",
}

ENGLISH_MARKERS = {
    "about",
    "advertising",
    "and",
    "are",
    "contact",
    "dollars",
    "email",
    "for",
    "from",
    "have",
    "how",
    "looking",
    "more",
    "results",
    "that",
    "the",
    "this",
    "thousands",
    "was",
    "with",
    "you",
}


@dataclass(frozen=True)
class HfSource:
    id: str
    dataset: str
    split: str
    text_fields: tuple[str, ...]
    config: str | None = None
    license: str = "unknown"
    notes: str = ""
    max_rows: int | None = None


@dataclass(frozen=True)
class LocalSource:
    id: str
    path: Path
    license: str
    notes: str = ""


@dataclass
class SourceStats:
    id: str
    status: str
    license: str
    notes: str
    rows_seen: int = 0
    lines_written: int = 0
    words_written: int = 0
    skipped_empty: int = 0
    skipped_non_kinyarwanda: int = 0
    skipped_duplicate: int = 0
    error: str | None = None


HF_SOURCES: tuple[HfSource, ...] = (
    HfSource(
        id="mbaza-monolingual-v01.1",
        dataset="mbazaNLP/kinyarwanda_monolingual_v01.1",
        split="train",
        text_fields=("text",),
        license="cc-by-4.0",
        notes="Large preferred version; gated on Hugging Face, so this may fail without access.",
    ),
    HfSource(
        id="mbaza-monolingual-v01.0",
        dataset="mbazaNLP/kinyarwanda_monolingual_v01.0",
        split="train",
        text_fields=("text",),
        license="cc-by-4.0",
        notes="Ungated Mbaza monolingual corpus; card reports about 25M words.",
    ),
    HfSource(
        id="rogerb-wikipedia-20230920",
        dataset="RogerB/Kinyarwanda_wikipedia20230920",
        split="train",
        text_fields=("text",),
        license="wikipedia-derived; attribution/share-alike review required",
        notes="Kinyarwanda Wikipedia snapshot.",
    ),
    HfSource(
        id="castorini-afriberta-gahuza",
        dataset="castorini/afriberta-corpus",
        config="gahuza",
        split="train",
        text_fields=("text",),
        license="apache-2.0",
        notes="AfriBERTa corpus; may be unavailable in datasets>=5 because it uses a legacy script.",
    ),
    HfSource(
        id="masakhane-mafand-en-kin",
        dataset="masakhane/mafand",
        config="en-kin",
        split="train",
        text_fields=("translation.kin", "translation.rw", "translation.tgt"),
        license="cc-by-nc-4.0",
        notes="Masakhane MAFAND news translation data; may be unavailable in datasets>=5.",
    ),
)

LOCAL_SOURCES: tuple[LocalSource, ...] = (
    LocalSource(
        id="digital-umuganda-tts-rw",
        path=ROOT / "data" / "approved" / "digital_umuganda_tts_rw.txt",
        license="cc0-1.0",
        notes="Approved local import from Digital Umuganda TTS sentence text.",
    ),
    LocalSource(
        id="digital-umuganda-mt-rw",
        path=ROOT / "data" / "approved" / "digital_umuganda_mt_rw.txt",
        license="cc-by-4.0",
        notes="Approved local import of Kinyarwanda side of Digital Umuganda MT.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-words", type=int, default=50_000_000)
    parser.add_argument("--min-words", type=int, default=10_000_000)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--min-line-chars", type=int, default=40)
    parser.add_argument("--max-line-chars", type=int, default=2_500)
    parser.add_argument("--allow-under-min", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    seen_hashes: set[str] = set()
    source_stats: list[SourceStats] = []
    total_words = 0
    total_lines = 0

    with args.out.open("w", encoding="utf-8") as handle:
        for source in LOCAL_SOURCES:
            stats, total_words, total_lines = consume_local_source(
                source,
                handle=handle,
                seen_hashes=seen_hashes,
                total_words=total_words,
                total_lines=total_lines,
                args=args,
            )
            source_stats.append(stats)
            if total_words >= args.target_words:
                break

        if total_words < args.target_words:
            for source in HF_SOURCES:
                stats, total_words, total_lines = consume_hf_source(
                    source,
                    handle=handle,
                    seen_hashes=seen_hashes,
                    total_words=total_words,
                    total_lines=total_lines,
                    args=args,
                )
                source_stats.append(stats)
                if total_words >= args.target_words:
                    break

    report = {
        "output": str(args.out),
        "target_words": args.target_words,
        "min_words": args.min_words,
        "total_words": total_words,
        "total_lines": total_lines,
        "elapsed_seconds": round(time.time() - started, 3),
        "sources": [asdict(stats) for stats in source_stats],
    }
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote={args.out}")
    print(f"report={args.report}")
    print(f"total_words={total_words}")
    print(f"total_lines={total_lines}")
    if total_words < args.min_words and not args.allow_under_min:
        print(
            f"ERROR: corpus below minimum word target: {total_words} < {args.min_words}",
            file=sys.stderr,
        )
        return 2
    return 0


def consume_local_source(
    source: LocalSource,
    *,
    handle,
    seen_hashes: set[str],
    total_words: int,
    total_lines: int,
    args: argparse.Namespace,
) -> tuple[SourceStats, int, int]:
    stats = SourceStats(
        id=source.id,
        status="ok" if source.path.exists() else "missing",
        license=source.license,
        notes=source.notes,
    )
    if not source.path.exists():
        stats.error = f"missing local source: {source.path}"
        return stats, total_words, total_lines

    for raw_line in source.path.read_text(encoding="utf-8").splitlines():
        stats.rows_seen += 1
        for line in clean_to_lines(
            raw_line,
            min_chars=args.min_line_chars,
            max_chars=args.max_line_chars,
        ):
            total_words, total_lines = maybe_write_line(
                line,
                handle=handle,
                seen_hashes=seen_hashes,
                stats=stats,
                total_words=total_words,
                total_lines=total_lines,
            )
            if total_words >= args.target_words:
                return stats, total_words, total_lines
    return stats, total_words, total_lines


def consume_hf_source(
    source: HfSource,
    *,
    handle,
    seen_hashes: set[str],
    total_words: int,
    total_lines: int,
    args: argparse.Namespace,
) -> tuple[SourceStats, int, int]:
    stats = SourceStats(
        id=source.id,
        status="ok",
        license=source.license,
        notes=source.notes,
    )
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            source.dataset,
            source.config,
            split=source.split,
            streaming=True,
        )
        for row in dataset:
            stats.rows_seen += 1
            if args.max_source_rows and stats.rows_seen > args.max_source_rows:
                break
            if source.max_rows and stats.rows_seen > source.max_rows:
                break
            for value in extract_row_texts(row, source.text_fields):
                for line in clean_to_lines(
                    value,
                    min_chars=args.min_line_chars,
                    max_chars=args.max_line_chars,
                ):
                    total_words, total_lines = maybe_write_line(
                        line,
                        handle=handle,
                        seen_hashes=seen_hashes,
                        stats=stats,
                        total_words=total_words,
                        total_lines=total_lines,
                    )
                    if total_words >= args.target_words:
                        return stats, total_words, total_lines
    except Exception as error:  # noqa: BLE001 - source failures should not abort.
        stats.status = "failed"
        stats.error = f"{type(error).__name__}: {error}"
    return stats, total_words, total_lines


def extract_row_texts(row: dict[str, object], fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for field in fields:
        current: object = row
        for part in field.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, str) and current.strip():
            values.append(current)
    return values


def clean_to_lines(text: str, *, min_chars: int, max_chars: int) -> list[str]:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\ufffd", " ")
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    text = MULTI_NEWLINE_RE.sub("\n\n", text).strip()
    if not text:
        return []

    lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if len(paragraph) < min_chars:
            continue
        if not looks_like_kinyarwanda_text(paragraph):
            continue
        while len(paragraph) > max_chars:
            split_at = paragraph.rfind(" ", 0, max_chars)
            if split_at < min_chars:
                split_at = max_chars
            chunk = paragraph[:split_at].strip()
            if len(chunk) >= min_chars and looks_like_kinyarwanda_text(chunk):
                lines.append(chunk)
            paragraph = paragraph[split_at:].strip()
        if len(paragraph) >= min_chars and looks_like_kinyarwanda_text(paragraph):
            lines.append(paragraph)
    return lines


def looks_like_kinyarwanda_text(text: str) -> bool:
    letters = 0
    latin_letters = 0
    rejected = 0
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category.startswith("L"):
            letters += 1
            name = unicodedata.name(char, "")
            if "LATIN" in name:
                latin_letters += 1
            continue
        if category[0] in {"N", "P"}:
            continue
        rejected += 1

    non_space = max(1, sum(not char.isspace() for char in text))
    if rejected / non_space > 0.08:
        return False
    if letters < 10:
        return False
    if latin_letters / max(1, letters) < 0.85:
        return False

    words = [word.lower().strip("’'") for word in WORD_RE.findall(text)]
    if len(words) >= 4:
        kinya_hits = sum(word in KINYA_MARKERS for word in words)
        english_hits = sum(word in ENGLISH_MARKERS for word in words)
        if kinya_hits == 0:
            return False
        if english_hits >= 3 and kinya_hits < 2:
            return False
    return True


def maybe_write_line(
    line: str,
    *,
    handle,
    seen_hashes: set[str],
    stats: SourceStats,
    total_words: int,
    total_lines: int,
) -> tuple[int, int]:
    if not line:
        stats.skipped_empty += 1
        return total_words, total_lines

    digest = hashlib.sha1(line.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        stats.skipped_duplicate += 1
        return total_words, total_lines
    seen_hashes.add(digest)

    words = len(line.split())
    handle.write(line + "\n")
    stats.lines_written += 1
    stats.words_written += words
    return total_words + words, total_lines + 1


if __name__ == "__main__":
    sys.exit(main())
