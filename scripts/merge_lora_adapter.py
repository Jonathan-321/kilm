"""Merge a LoRA adapter into a base causal LM."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = choose_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    model = load_model(args.base_model, dtype=dtype, device_map=args.device_map)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"merged_model={args.output_dir}")
    return 0


def choose_dtype(requested: str) -> torch.dtype:
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(model_name: str, *, dtype: torch.dtype, device_map: str):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )


if __name__ == "__main__":
    sys.exit(main())
