"""Generate text from a Hugging Face KILM checkpoint."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "kilm-llama-100m" / "checkpoint-50000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to generate from. May be passed more than once.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional UTF-8 file with one prompt per non-empty line.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = collect_prompts(args)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, dtype=dtype).to(device)
    model.eval()

    print(f"checkpoint={args.checkpoint}")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"prompts={len(prompts)}")

    for prompt in prompts:
        print(f"\n===== prompt: {prompt} =====")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **generation_kwargs(args, tokenizer),
            )
        for index, sequence in enumerate(generated, start=1):
            if args.num_return_sequences > 1:
                print(f"\n--- sample {index} ---")
            print(tokenizer.decode(sequence, skip_special_tokens=True))
    return 0


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        prompts.append("Muraho")
    return prompts


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


def load_model(checkpoint: Path, *, dtype: torch.dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(str(checkpoint), torch_dtype=dtype)


def generation_kwargs(args: argparse.Namespace, tokenizer) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_return_sequences": args.num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        kwargs["do_sample"] = False
        return kwargs

    kwargs.update(
        {
            "do_sample": True,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        }
    )
    if args.top_k > 0:
        kwargs["top_k"] = args.top_k
    if args.top_p < 1.0:
        kwargs["top_p"] = args.top_p
    return kwargs


if __name__ == "__main__":
    sys.exit(main())
