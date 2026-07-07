"""Prepare Kinyarwanda conversation data for supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import csv
import html
import json
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "sft" / "processed"

PAIR_KEY_SETS = [
    ("user", "assistant"),
    ("prompt", "response"),
    ("instruction", "output"),
    ("question", "answer"),
    ("input", "target"),
]
ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class PreparedConversation:
    item_id: str
    source: str
    messages: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "json", "csv", "tsv"),
        default="auto",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-user-chars", type=int, default=2)
    parser.add_argument("--min-assistant-chars", type=int, default=2)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be >=0 and <1")

    conversations: list[PreparedConversation] = []
    skipped = 0
    for path in args.input:
        for index, payload in enumerate(iter_records(path, args.format), start=1):
            prepared = normalize_record(
                payload,
                item_id=str(payload.get("id") or payload.get("uuid") or f"{path.stem}-{index}"),
                source=str(payload.get("source") or path.name),
                system_prompt=args.system_prompt,
                min_user_chars=args.min_user_chars,
                min_assistant_chars=args.min_assistant_chars,
            )
            if prepared is None:
                skipped += 1
                continue
            conversations.append(prepared)

    if not conversations:
        raise RuntimeError("No valid conversations found.")

    if not args.keep_duplicates:
        conversations = dedupe(conversations)

    rng = random.Random(args.seed)
    rng.shuffle(conversations)
    validation_count = int(round(len(conversations) * args.validation_fraction))
    if args.validation_fraction > 0 and len(conversations) > 1:
        validation_count = max(1, validation_count)
    validation = conversations[:validation_count]
    train = conversations[validation_count:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)

    summary = {
        "inputs": [str(path) for path in args.input],
        "train_examples": len(train),
        "validation_examples": len(validation),
        "skipped_records": skipped,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "system_prompt": args.system_prompt,
        "schema": {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def iter_records(path: Path, requested_format: str) -> list[dict[str, Any]]:
    data_format = detect_format(path, requested_format)
    if data_format == "jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL") from error
        return records
    if data_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        raise ValueError(f"{path}: expected a JSON list or an object with data=[]")
    if data_format in {"csv", "tsv"}:
        delimiter = "\t" if data_format == "tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise AssertionError(f"unsupported format: {data_format}")


def detect_format(path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".tab"}:
        return "tsv"
    raise ValueError(f"Cannot infer format for {path}; pass --format explicitly.")


def normalize_record(
    payload: dict[str, Any],
    *,
    item_id: str,
    source: str,
    system_prompt: str,
    min_user_chars: int,
    min_assistant_chars: int,
) -> PreparedConversation | None:
    messages: list[dict[str, str]] | None = None
    if isinstance(payload.get("messages"), list):
        messages = normalize_messages(payload["messages"])
    else:
        messages = pair_to_messages(payload, system_prompt=system_prompt)

    if not messages:
        return None
    if system_prompt and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": normalize_text(system_prompt)}] + messages

    user_chars = sum(len(message["content"]) for message in messages if message["role"] == "user")
    assistant_chars = sum(
        len(message["content"]) for message in messages if message["role"] == "assistant"
    )
    if user_chars < min_user_chars or assistant_chars < min_assistant_chars:
        return None
    if messages[-1]["role"] != "assistant":
        return None

    return PreparedConversation(item_id=item_id, source=source, messages=messages)


def normalize_messages(raw_messages: list[Any]) -> list[dict[str, str]] | None:
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        role = str(raw.get("role", "")).strip().lower()
        content = normalize_text(raw.get("content", ""))
        if role not in ALLOWED_ROLES or not content:
            continue
        messages.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in messages):
        return None
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def pair_to_messages(payload: dict[str, Any], *, system_prompt: str) -> list[dict[str, str]] | None:
    user_text = ""
    assistant_text = ""
    for user_key, assistant_key in PAIR_KEY_SETS:
        user_text = normalize_text(payload.get(user_key, ""))
        assistant_text = normalize_text(payload.get(assistant_key, ""))
        if user_text and assistant_text:
            break
    if not user_text or not assistant_text:
        return None

    messages: list[dict[str, str]] = []
    system_text = normalize_text(payload.get("system", "") or system_prompt)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    )
    return messages


def normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe(conversations: list[PreparedConversation]) -> list[PreparedConversation]:
    seen: set[str] = set()
    unique: list[PreparedConversation] = []
    for conversation in conversations:
        key = json.dumps(conversation.messages, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(conversation)
    return unique


def write_jsonl(path: Path, conversations: list[PreparedConversation]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            payload = {
                "id": conversation.item_id,
                "source": conversation.source,
                "messages": conversation.messages,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
