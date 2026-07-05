"""Text cleanup and split helpers for KILM corpora."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from kilm.corpus import CorpusRecord


@dataclass(frozen=True)
class PreparedCorpus:
    lines: list[str]
    stats: dict[str, int | bool]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def prepare_lines(
    text: str,
    *,
    min_line_chars: int = 1,
    dedupe: bool = True,
) -> PreparedCorpus:
    if min_line_chars < 1:
        raise ValueError("min_line_chars must be at least 1")

    raw_lines = text.splitlines()
    lines: list[str] = []
    seen: set[str] = set()
    blank_count = 0
    short_count = 0
    duplicate_count = 0

    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            blank_count += 1
            continue
        if len(line) < min_line_chars:
            short_count += 1
            continue
        if dedupe and line in seen:
            duplicate_count += 1
            continue
        seen.add(line)
        lines.append(line)

    if not lines:
        raise ValueError("no usable lines after corpus preparation")

    return PreparedCorpus(
        lines=lines,
        stats={
            "raw_lines": len(raw_lines),
            "prepared_lines": len(lines),
            "removed_blank_lines": blank_count,
            "removed_short_lines": short_count,
            "removed_duplicate_lines": duplicate_count,
            "dedupe": dedupe,
            "min_line_chars": min_line_chars,
        },
    )


def split_lines(
    lines: list[str],
    *,
    val_fraction: float = 0.1,
    seed: int = 1337,
) -> tuple[list[str], list[str]]:
    if len(lines) < 2:
        raise ValueError("need at least two lines to create a train/val split")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    indexed_lines = list(enumerate(lines))
    Random(seed).shuffle(indexed_lines)
    val_count = max(1, round(len(lines) * val_fraction))
    val_count = min(val_count, len(lines) - 1)
    val_indexes = {idx for idx, _ in indexed_lines[:val_count]}

    train = [line for idx, line in enumerate(lines) if idx not in val_indexes]
    val = [line for idx, line in enumerate(lines) if idx in val_indexes]
    return train, val


def render_corpus_card(
    *,
    source: CorpusRecord,
    prepared: PreparedCorpus,
    train_lines: list[str],
    val_lines: list[str],
    val_fraction: float,
    seed: int,
) -> str:
    stats = prepared.stats
    lines = [
        "# Corpus Card",
        "",
        "## Source",
        "",
        f"- ID: `{source.id}`",
        f"- Status: `{source.status}`",
        f"- Path: `{source.path}`",
        f"- Source: {source.source or 'n/a'}",
        f"- License: {source.license or 'n/a'}",
        f"- Notes: {source.notes or 'n/a'}",
        "",
        "## Preparation",
        "",
        f"- Raw lines: `{stats['raw_lines']}`",
        f"- Prepared lines: `{stats['prepared_lines']}`",
        f"- Removed blank lines: `{stats['removed_blank_lines']}`",
        f"- Removed short lines: `{stats['removed_short_lines']}`",
        f"- Removed duplicate lines: `{stats['removed_duplicate_lines']}`",
        f"- Minimum line characters: `{stats['min_line_chars']}`",
        f"- Dedupe: `{stats['dedupe']}`",
        "",
        "## Split",
        "",
        f"- Train lines: `{len(train_lines)}`",
        f"- Validation lines: `{len(val_lines)}`",
        f"- Validation fraction: `{val_fraction}`",
        f"- Seed: `{seed}`",
        "",
        "## Use",
        "",
        "Use this prepared corpus only within the status allowed by the source "
        "record. A `toy` source is still not approved model-training data.",
    ]
    return "\n".join(lines).rstrip() + "\n"
