"""Write the final KILM training report."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "checkpoints" / "kilm-llama-100m"
DEFAULT_DATASET = ROOT / "data" / "tokenized" / "kinyarwanda_spm_1024"
DEFAULT_CORPUS_REPORT = ROOT / "data" / "kinyarwanda_full_corpus_stats.json"
DEFAULT_OUT = ROOT / "docs" / "FINAL_RUN_REPORT.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus-report", type=Path, default=DEFAULT_CORPUS_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prompt", default="Muraho")
    parser.add_argument("--sample-tokens", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus_report = read_json(args.corpus_report)
    token_report = read_json(args.dataset_dir / "tokenization_report.json")
    train_metrics = read_json(args.model_dir / "train_results.json")
    eval_metrics = read_json(args.model_dir / "eval_results.json")
    metadata = read_json(args.model_dir / "training_metadata.json")
    training_args = metadata.get("training_args", {})
    sources = corpus_report.get("sources", [])

    greedy_samples: list[str] = []
    temp_samples: list[str] = []
    if args.model_dir.exists():
        greedy_samples, temp_samples = generate_samples(args)

    lines = [
        "# Final KILM Run Report",
        "",
        "## Dataset",
        "",
        f"- Corpus path: `{corpus_report.get('output', 'n/a')}`",
        f"- Corpus words: `{corpus_report.get('total_words', 'n/a')}`",
        f"- Corpus lines: `{corpus_report.get('total_lines', 'n/a')}`",
        f"- Tokenized train tokens: `{token_report.get('train_tokens', 'n/a')}`",
        f"- Tokenized validation tokens: `{token_report.get('validation_tokens', 'n/a')}`",
        f"- Total tokenized tokens: `{token_report.get('total_tokens', 'n/a')}`",
        "- Split: `98% train / 2% validation`",
        "",
        "## Sources",
        "",
    ]
    lines.extend(format_sources(sources))
    lines.extend([
        "",
        "## Model",
        "",
        f"- Model directory: `{args.model_dir}`",
        f"- Parameters: `{metadata.get('parameter_count', 'n/a')}`",
        "- Architecture: LLaMA-style causal LM, 12 layers, 12 attention heads, hidden size 768, context length 1024",
        "- Tokenizer: SentencePiece BPE, 32000 vocabulary size, byte fallback enabled",
        "",
        "## Run Scope",
        "",
        "- Script default: `50000` max steps",
        f"- Completed local baseline: `{get_nested(training_args, 'max_steps', 'n/a')}` steps",
        "- Reason: local Apple MPS compute can run the pipeline end to end, but a 50000-step run would take substantially longer than this session.",
        f"- Checkpoint/eval/sample interval used: `{get_nested(training_args, 'save_steps', 'n/a')}` steps",
        "- Mixed precision: disabled on local MPS; CUDA runs use bf16/fp16 when available.",
        "",
        "## Metrics",
        "",
        f"- Final train loss: `{train_metrics.get('train_loss', 'n/a')}`",
        f"- Final train perplexity: `{train_metrics.get('train_perplexity', 'n/a')}`",
        f"- Final validation loss: `{eval_metrics.get('eval_loss', 'n/a')}`",
        f"- Final validation perplexity: `{eval_metrics.get('eval_perplexity', 'n/a')}`",
        "",
        "## Greedy Samples",
        "",
    ])
    lines.extend(format_samples(greedy_samples))
    lines.extend(["", "## Temperature 0.7 / Top-k 40 Samples", ""])
    lines.extend(format_samples(temp_samples))
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def generate_samples(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir))
    device = select_device()
    model.to(device)
    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    greedy: list[str] = []
    sampled: list[str] = []
    for _ in range(5):
        with torch.no_grad():
            greedy_ids = model.generate(
                **inputs,
                max_new_tokens=args.sample_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sampled_ids = model.generate(
                **inputs,
                max_new_tokens=args.sample_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        greedy.append(tokenizer.decode(greedy_ids[0], skip_special_tokens=True))
        sampled.append(tokenizer.decode(sampled_ids[0], skip_special_tokens=True))
    return greedy, sampled


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_samples(samples: list[str]) -> list[str]:
    if not samples:
        return ["No samples available."]
    lines: list[str] = []
    for idx, sample in enumerate(samples, start=1):
        lines.extend([f"### Sample {idx}", "", "```text", sample.strip(), "```", ""])
    return lines


def get_nested(payload: object, key: str, default: object) -> object:
    if not isinstance(payload, dict):
        return default
    return payload.get(key, default)


def format_sources(sources: object) -> list[str]:
    if not isinstance(sources, list) or not sources:
        return ["No source stats available."]

    lines: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = source.get("id", "unknown")
        status = source.get("status", "unknown")
        words = source.get("words_written", 0)
        notes = source.get("notes", "")
        if status == "ok":
            lines.append(f"- {source_id}: `{words}` cleaned words")
        else:
            lines.append(f"- {source_id}: not included (`{status}`; {notes})")
    return lines or ["No source stats available."]


if __name__ == "__main__":
    sys.exit(main())
