"""Train a 100M-class Kinyarwanda causal LM from scratch."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "tokenized" / "kinyarwanda_spm_1024"
DEFAULT_TOKENIZER = ROOT / "tokenizer"
DEFAULT_CONFIG = ROOT / "configs" / "llama_100m.json"
DEFAULT_OUT = ROOT / "checkpoints" / "kilm-llama-100m"
DEFAULT_LOGS = ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=2_000)
    parser.add_argument("--eval-steps", type=int, default=2_000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=2_000)
    parser.add_argument("--sample-prompt", default="Muraho")
    parser.add_argument("--sample-new-tokens", type=int, default=180)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--oom-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_batch_size = args.per_device_train_batch_size
    last_error: Exception | None = None
    for attempt in range(args.oom_retries + 1):
        try:
            run_training(args, train_batch_size=train_batch_size)
            return 0
        except RuntimeError as error:
            last_error = error
            if not is_oom_error(error) or train_batch_size <= 1:
                raise
            train_batch_size = max(1, train_batch_size // 2)
            print(
                "OOM detected; retrying with "
                f"per_device_train_batch_size={train_batch_size}",
                flush=True,
            )
            clear_device_cache()
    if last_error:
        raise last_error
    return 1


def run_training(args: argparse.Namespace, *, train_batch_size: int) -> None:
    from datasets import load_from_disk

    dataset = load_from_disk(str(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    config_payload["vocab_size"] = len(tokenizer)
    config = LlamaConfig(**config_payload)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config)
    model.config.use_cache = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_config.json").write_text(
        model.config.to_json_string() + "\n",
        encoding="utf-8",
    )

    bf16, fp16 = choose_precision()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        do_train=True,
        do_eval=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=args.logging_steps,
        disable_tqdm=args.disable_tqdm,
        report_to=[],
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=args.seed,
    )

    metadata = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["test"]),
        "train_tokens": len(dataset["train"]) * config.max_position_embeddings,
        "validation_tokens": len(dataset["test"]) * config.max_position_embeddings,
        "training_args": training_args.to_dict(),
        "precision": {"bf16": bf16, "fp16": fp16},
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"parameters={metadata['parameter_count']}")
    print(f"train_tokens={metadata['train_tokens']}")
    print(f"validation_tokens={metadata['validation_tokens']}")
    print(f"bf16={bf16} fp16={fp16}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            SampleCallback(
                tokenizer=tokenizer,
                logs_dir=args.logs_dir,
                interval=args.sample_steps,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_new_tokens,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
            )
        ],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    metrics = train_result.metrics
    metrics["train_perplexity"] = safe_perplexity(metrics.get("train_loss"))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_perplexity"] = safe_perplexity(eval_metrics.get("eval_loss"))
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


class SampleCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        logs_dir: Path,
        interval: int,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.logs_dir = logs_dir
        self.interval = interval
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.path = logs_dir / "samples.txt"

    def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
        if model is None or self.interval <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return control

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        sample = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== step {state.global_step} =====\n")
            handle.write(sample.strip() + "\n")
        if was_training:
            model.train()
        return control


def choose_precision() -> tuple[bool, bool]:
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported(), not torch.cuda.is_bf16_supported()
    return False, False


def clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def safe_perplexity(loss: object) -> float | None:
    if loss is None:
        return None
    value = float(loss)
    if value > 50:
        return float("inf")
    return math.exp(value)


if __name__ == "__main__":
    sys.exit(main())
