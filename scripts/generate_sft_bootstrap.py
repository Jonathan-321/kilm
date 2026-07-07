"""Build a bootstrap Kinyarwanda SFT dataset from open HF sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import random
import re
import sys

from datasets import load_dataset
from huggingface_hub import list_datasets


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "sft" / "raw_conversations.jsonl"
DEFAULT_BENCHMARK = ROOT / "data" / "eval" / "kinyarwanda_conversation_benchmark.jsonl"
RW_KEYS = ("rw", "kin", "kinyarwanda", "target_rw", "sentence_rw")
OTHER_KEYS = ("en", "eng", "english", "fr", "fra", "french")
SEED_DATASETS = [
    "ChrisToukmaji/kinyarwanda_instruction_tuning",
    "Rwanda-Tech-Project/kinyarwanda-instruction-dataset",
    "Mikecyane/Kinyarwanda_chat",
    "souvorinkg/english_kinyarwanda",
    "mbazaNLP/common-voice-kinyarwanda-english-dataset",
    "DigitalUmuganda/kinyarwanda-english-machine-translation-dataset",
    "mbazaNLP/Kinyarwanda_English_parallel_dataset",
    "masakhane/mafand",
    "mbazaNLP/mbazaNMT",
    "mbazaNLP/kinyarwanda-english",
    "mbazaNLP/NMT_Turku-to-English",
]
CHAT_LINE_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+.*?\s+-\s+([^:]+):\s+(.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--min-examples", type=int, default=2_000)
    parser.add_argument("--max-examples", type=int, default=20_000)
    parser.add_argument("--benchmark-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--search-limit", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    candidates = candidate_datasets(search_limit=args.search_limit)

    for dataset_id in candidates:
        if len(examples) >= args.max_examples:
            break
        try:
            before = len(examples)
            remaining = args.max_examples - len(examples)
            examples.extend(load_dataset_examples(dataset_id, remaining=remaining))
            added = len(examples) - before
            print(f"dataset={dataset_id} added={added}", flush=True)
        except Exception as error:
            errors.append(f"{dataset_id}: {error}")
            print(f"dataset={dataset_id} failed={error}", flush=True)

    examples = dedupe(examples)
    rng.shuffle(examples)
    if len(examples) < args.min_examples:
        print(
            f"warning: only found {len(examples)} examples, below requested {args.min_examples}",
            flush=True,
        )

    benchmark = examples[: args.benchmark_size]
    trainish = examples[args.benchmark_size :]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, trainish)
    write_benchmark(args.benchmark_out, benchmark)
    summary = {
        "raw_conversations": str(args.out),
        "benchmark": str(args.benchmark_out),
        "conversation_examples": len(trainish),
        "benchmark_examples": len(benchmark),
        "errors": errors,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def candidate_datasets(*, search_limit: int) -> list[str]:
    candidates: list[str] = []
    for dataset_id in SEED_DATASETS:
        if dataset_id not in candidates:
            candidates.append(dataset_id)
    for query in ("kinyarwanda", "rw en translation", "masakhane kinyarwanda"):
        try:
            for info in list_datasets(search=query, limit=search_limit):
                dataset_id = info.id
                if dataset_id not in candidates:
                    candidates.append(dataset_id)
        except Exception as error:
            print(f"dataset_search_failed query={query}: {error}", flush=True)
    return candidates


def load_dataset_examples(dataset_id: str, *, remaining: int) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_id)
    rows: list[dict[str, Any]] = []
    for split_name, split in dataset.items():
        chat_turns: list[tuple[str, str]] = []
        for row in split:
            instruction = extract_instruction(row, dataset_id, split_name)
            if instruction is not None:
                rows.append(instruction)
                if len(rows) >= remaining:
                    return rows[:remaining]

            chat_turn = extract_chat_turn(row)
            if chat_turn is not None:
                chat_turns.append(chat_turn)
                if len(chat_turns) >= 2:
                    rows.extend(chat_pairs(chat_turns, dataset_id, split_name))
                    chat_turns = chat_turns[-1:]
                if len(rows) >= remaining:
                    return rows[:remaining]

            pair = extract_pair(row)
            if pair is None:
                continue
            rw_text, other_text, other_lang = pair
            rows.extend(conversation_pairs(rw_text, other_text, other_lang, dataset_id, split_name))
            if len(rows) >= remaining:
                return rows[:remaining]
    return rows


def extract_instruction(
    row: dict[str, Any],
    dataset_id: str,
    split_name: str,
) -> dict[str, Any] | None:
    instruction = clean(row.get("translated_instruction") or row.get("instruction"))
    response = clean(row.get("translated_output") or row.get("response") or row.get("output"))
    extra_input = clean(row.get("translated_input") or row.get("input") or row.get("context"))
    if not instruction or not response:
        return None
    if not good_text(instruction) or not good_text(response):
        return None
    if extra_input and extra_input.lower() not in {"nan", "none", "null"}:
        user_content = f"{instruction}\n\n{extra_input}"
    else:
        user_content = instruction
    return {
        "source": f"{dataset_id}:{split_name}",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ],
    }


def extract_chat_turn(row: dict[str, Any]) -> tuple[str, str] | None:
    text = clean(row.get("text", ""))
    match = CHAT_LINE_RE.match(text)
    if not match:
        return None
    speaker = match.group(1).strip()
    message = match.group(2).strip()
    lower = message.lower()
    skip_fragments = (
        "messages and calls are end-to-end encrypted",
        "waiting for this message",
        "media omitted",
        "missed voice call",
        "deleted this message",
        "this message was deleted",
    )
    if any(fragment in lower for fragment in skip_fragments):
        return None
    if not good_text(message):
        return None
    return speaker, message


def chat_pairs(
    turns: list[tuple[str, str]],
    dataset_id: str,
    split_name: str,
) -> list[dict[str, Any]]:
    if len(turns) < 2:
        return []
    previous_speaker, previous_message = turns[-2]
    speaker, message = turns[-1]
    if previous_speaker == speaker:
        return []
    return [
        {
            "source": f"{dataset_id}:{split_name}",
            "messages": [
                {"role": "user", "content": previous_message},
                {"role": "assistant", "content": message},
            ],
        }
    ]


def extract_pair(row: dict[str, Any]) -> tuple[str, str, str] | None:
    translation = row.get("translation")
    if isinstance(translation, dict):
        rw_text = first_value(translation, RW_KEYS)
        other_key, other_text = first_key_value(translation, OTHER_KEYS)
        if rw_text and other_text:
            return clean(rw_text), clean(other_text), normalize_lang(other_key)

    rw_text = first_value(row, RW_KEYS)
    other_key, other_text = first_key_value(row, OTHER_KEYS)
    if rw_text and other_text:
        return clean(rw_text), clean(other_text), normalize_lang(other_key)

    source = row.get("source") or row.get("src") or row.get("text")
    target = row.get("target") or row.get("tgt") or row.get("label")
    if source and target:
        source_text = clean(source)
        target_text = clean(target)
        source_lang = str(row.get("source_lang") or row.get("src_lang") or "").lower()
        target_lang = str(row.get("target_lang") or row.get("tgt_lang") or "").lower()
        if "rw" in source_lang or "kin" in source_lang:
            return source_text, target_text, normalize_lang(target_lang or "en")
        if "rw" in target_lang or "kin" in target_lang:
            return target_text, source_text, normalize_lang(source_lang or "en")
    return None


def conversation_pairs(
    rw_text: str,
    other_text: str,
    other_lang: str,
    dataset_id: str,
    split_name: str,
) -> list[dict[str, Any]]:
    if not good_text(rw_text) or not good_text(other_text):
        return []
    other_name = "Icyongereza" if other_lang == "en" else "Igifaransa"
    return [
        {
            "source": f"{dataset_id}:{split_name}",
            "messages": [
                {"role": "user", "content": f"Hindura mu {other_name}: {rw_text}"},
                {"role": "assistant", "content": other_text},
            ],
        },
        {
            "source": f"{dataset_id}:{split_name}",
            "messages": [
                {"role": "user", "content": f"Hindura mu Kinyarwanda: {other_text}"},
                {"role": "assistant", "content": rw_text},
            ],
        },
    ]


def first_value(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in payload and payload[key]:
            return str(payload[key])
    return ""


def first_key_value(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str]:
    for key in keys:
        if key in payload and payload[key]:
            return key, str(payload[key])
    return "", ""


def normalize_lang(value: str) -> str:
    value = value.lower()
    if value.startswith("fr") or "french" in value:
        return "fr"
    return "en"


def clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or ""))
    return text.strip()


def good_text(text: str) -> bool:
    if len(text) < 4 or len(text) > 800:
        return False
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(1, len(text)) > 0.45


def dedupe(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for example in examples:
        key = json.dumps(example["messages"], sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            payload = {"id": f"sft-{index:06d}", **row}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            messages = row["messages"]
            payload = {
                "id": f"bench-{index:04d}",
                "category": "translation",
                "source": row.get("source", ""),
                "messages": messages,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
