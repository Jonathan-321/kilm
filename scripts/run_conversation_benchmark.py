"""Run a Kinyarwanda conversation benchmark and write review files."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "data" / "eval" / "kinyarwanda_conversation_benchmark.jsonl"
DEFAULT_OUT_DIR = ROOT / "logs" / "benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--adapter", help="Optional LoRA adapter path for --model.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = read_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise RuntimeError("No benchmark items found.")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    tokenizer_name = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, adapter=args.adapter, dtype=dtype).to(device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.out_dir / f"conversation_benchmark_{stamp}.jsonl"
    review_path = args.out_dir / f"conversation_benchmark_review_{stamp}.tsv"

    rows = []
    for item in items:
        prompt_messages, reference = prompt_from_item(item)
        prompt_text = format_messages(
            prompt_messages,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_k=args.top_k,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append(
            {
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "prompt": prompt_messages[-1]["content"],
                "reference": reference,
                "output": generated,
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_review_sheet(review_path, rows)
    print(f"wrote_jsonl={jsonl_path}")
    print(f"wrote_review_tsv={review_path}")
    return 0


def read_benchmark(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from error
    return rows


def prompt_from_item(item: dict) -> tuple[list[dict[str, str]], str]:
    if isinstance(item.get("messages"), list):
        messages = item["messages"]
        if messages and messages[-1].get("role") == "assistant":
            return messages[:-1], messages[-1].get("content", "")
        return messages, item.get("reference", "")
    prompt = item.get("prompt") or item.get("user") or item.get("question")
    if not prompt:
        raise ValueError(f"Benchmark item has no prompt-like field: {item}")
    return [{"role": "user", "content": str(prompt)}], str(item.get("reference", ""))


def write_review_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "id",
        "category",
        "prompt",
        "reference",
        "output",
        "fluency_1_5",
        "helpfulness_1_5",
        "correctness_1_5",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "fluency_1_5": "",
                    "helpfulness_1_5": "",
                    "correctness_1_5": "",
                    "notes": "",
                }
            )


def format_messages(
    messages: list[dict[str, str]],
    *,
    tokenizer,
    add_generation_prompt: bool,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = [f"{message['role'].title()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_model(model_name: str, *, adapter: str | None, dtype: torch.dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model


if __name__ == "__main__":
    sys.exit(main())
