"""Supervised fine-tuning with TRL SFTTrainer and QLoRA."""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import inspect
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "sft" / "processed" / "train.jsonl"
DEFAULT_VAL = ROOT / "data" / "sft" / "processed" / "validation.jsonl"
DEFAULT_OUT = ROOT / "checkpoints" / "sft" / "qwen2.5-7b-kinyarwanda-qlora"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_size = args.per_device_train_batch_size
    for attempt in range(args.oom_retries + 1):
        args.per_device_train_batch_size = batch_size
        try:
            run_training(args)
            return 0
        except RuntimeError as error:
            if not is_oom_error(error) or batch_size <= 1 or attempt >= args.oom_retries:
                raise
            batch_size = max(1, batch_size // 2)
            print(
                "OOM detected; retrying with "
                f"per_device_train_batch_size={batch_size}",
                flush=True,
            )
            clear_memory()
    return 1


def run_training(args: argparse.Namespace) -> None:
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training requires CUDA. "
            "Pass --no-load-in-4bit for CPU/MPS tests."
        )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dtype = choose_dtype(args.dtype)
    model_name, tokenizer, model = load_model_and_tokenizer(args, dtype=dtype)
    dataset = load_conversation_dataset(args)
    text_dataset = dataset.map(
        lambda row: {"text": format_messages(row["messages"], tokenizer=tokenizer)},
        remove_columns=dataset["train"].column_names,
        desc="Applying chat template",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = make_sft_config(
        args,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
    )

    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        lora_config=lora_config,
        train_dataset=text_dataset["train"],
        eval_dataset=text_dataset.get("validation"),
    )
    print(f"base_model={model_name}", flush=True)
    print(f"train_examples={len(text_dataset['train'])}", flush=True)
    if "validation" in text_dataset:
        print(f"validation_examples={len(text_dataset['validation'])}", flush=True)
    print(f"per_device_train_batch_size={args.per_device_train_batch_size}", flush=True)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


def load_conversation_dataset(args: argparse.Namespace):
    data_files = {"train": str(args.train_file)}
    if args.validation_file.exists():
        data_files["validation"] = str(args.validation_file)
    return load_dataset("json", data_files=data_files)


def load_model_and_tokenizer(args: argparse.Namespace, *, dtype: torch.dtype):
    last_error: Exception | None = None
    for model_name in [args.model_name, args.fallback_model]:
        if not model_name:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=args.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = load_model(args, model_name=model_name, dtype=dtype)
            model.config.use_cache = False
            return model_name, tokenizer, model
        except Exception as error:
            last_error = error
            print(f"failed_to_load_model={model_name}: {error}", flush=True)
            clear_memory()
    assert last_error is not None
    raise last_error


def load_model(args: argparse.Namespace, *, model_name: str, dtype: torch.dtype):
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


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


def format_messages(messages: list[dict[str, str]], *, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    parts: list[str] = []
    for message in messages:
        role = message["role"].strip().title()
        content = message["content"].strip()
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def make_sft_config(args: argparse.Namespace, *, bf16: bool, fp16: bool) -> SFTConfig:
    payload = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps" if args.validation_file.exists() else "no",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
        "max_length": args.max_seq_length,
        "packing": args.packing,
        "dataset_text_field": "text",
    }
    parameters = set(inspect.signature(SFTConfig).parameters)
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        payload["evaluation_strategy"] = payload.pop("eval_strategy")
    filtered = {key: value for key, value in payload.items() if key in parameters}
    return SFTConfig(**filtered)


def build_sft_trainer(
    *,
    model,
    tokenizer,
    training_args: SFTConfig,
    lora_config: LoraConfig,
    train_dataset,
    eval_dataset,
) -> SFTTrainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
        "processing_class": tokenizer,
    }
    try:
        return SFTTrainer(**kwargs)
    except TypeError:
        kwargs["tokenizer"] = kwargs.pop("processing_class")
        return SFTTrainer(**kwargs)


def is_oom_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas" in text


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
