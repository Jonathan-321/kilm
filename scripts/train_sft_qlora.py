"""Supervised fine-tuning with QLoRA on Kinyarwanda chat data."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "sft" / "processed" / "train.jsonl"
DEFAULT_VAL = ROOT / "data" / "sft" / "processed" / "validation.jsonl"
DEFAULT_OUT = ROOT / "checkpoints" / "sft" / "kinyarwanda-qlora"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
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
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training requires CUDA. "
            "Pass --no-load-in-4bit for CPU/MPS tests."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = choose_dtype(args.dtype)
    model = load_model(args, dtype=dtype)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_conversation_dataset(args)
    tokenized = dataset.map(
        lambda row: tokenize_row(row, tokenizer=tokenizer, max_length=args.max_seq_length),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing conversations",
    )

    training_args = make_training_args(
        args,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    return 0


def load_conversation_dataset(args: argparse.Namespace):
    data_files = {"train": str(args.train_file)}
    if args.validation_file.exists():
        data_files["validation"] = str(args.validation_file)
    return load_dataset("json", data_files=data_files)


def load_model(args: argparse.Namespace, *, dtype: torch.dtype):
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
        return AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, **kwargs)


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


def tokenize_row(row: dict, *, tokenizer, max_length: int) -> dict[str, list[int]]:
    text = format_messages(row["messages"], tokenizer=tokenizer, add_generation_prompt=False)
    encoded = tokenizer(text, max_length=max_length, truncation=True, padding=False)
    encoded["labels"] = list(encoded["input_ids"])
    return encoded


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
    parts: list[str] = []
    for message in messages:
        role = message["role"].strip().title()
        parts.append(f"{role}: {message['content'].strip()}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def make_training_args(args: argparse.Namespace, *, bf16: bool, fp16: bool) -> TrainingArguments:
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
    }
    try:
        return TrainingArguments(**payload)
    except TypeError:
        payload["evaluation_strategy"] = payload.pop("eval_strategy")
        return TrainingArguments(**payload)


if __name__ == "__main__":
    sys.exit(main())
