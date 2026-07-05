"""Tokenizer analysis helpers for KILM."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import re

from kilm.corpus import CorpusRecord
from kilm.tokenizers import AnyTokenizer


WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def summarize_tokenizer(
    name: str,
    tokenizer: AnyTokenizer,
    text: str,
    *,
    top_k: int = 20,
) -> dict[str, object]:
    token_ids = tokenizer.encode(text)
    token_payload = tokenizer.to_dict()
    vocab = list(token_payload["vocab"])
    tokens = [vocab[token_id] for token_id in token_ids]
    token_counts = Counter(tokens)
    word_count = count_words(text)

    return {
        "name": name,
        "vocab_size": tokenizer.vocab_size,
        "num_merges": len(token_payload["merges"]),
        "num_characters": len(text),
        "num_words": word_count,
        "num_tokens": len(token_ids),
        "tokens_per_character": round(len(token_ids) / len(text), 4),
        "tokens_per_word": round(len(token_ids) / max(1, word_count), 4),
        "characters_per_token": round(len(text) / max(1, len(token_ids)), 4),
        "top_tokens": [
            {"token": token, "count": count}
            for token, count in token_counts.most_common(top_k)
        ],
    }


def render_tokenizer_report(
    *,
    corpus: CorpusRecord,
    summaries: list[dict[str, object]],
) -> str:
    lines = [
        "# Tokenizer Analysis",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Corpus",
        "",
        f"- ID: `{corpus.id}`",
        f"- Status: `{corpus.status}`",
        f"- Path: `{corpus.path}`",
        f"- Description: {corpus.description or 'n/a'}",
        "",
        "## Summary",
        "",
        "| tokenizer | vocab | merges | chars | words | tokens | "
        "tokens/char | tokens/word | chars/token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {name} | {vocab_size} | {num_merges} | {num_characters} | "
            "{num_words} | {num_tokens} | {tokens_per_character} | "
            "{tokens_per_word} | {characters_per_token} |".format(**summary)
        )

    lines.extend(["", "## Top Tokens", ""])
    for summary in summaries:
        lines.append(f"### {summary['name']}")
        lines.append("")
        for item in summary["top_tokens"]:
            token = str(item["token"]).replace("\n", "\\n")
            lines.append(f"- `{token}`: {item['count']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
