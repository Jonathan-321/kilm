"""Run the Track A sandbox end to end on a toy corpus."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from kilm.checkpointing import load_checkpoint
from kilm.configs import build_model_config
from kilm.corpus import load_corpus_text
from kilm.reporting import render_run_report
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from kilm.tokenizers import AnyTokenizer, tokenizer_from_dict, train_tokenizer
from kilm.training import (
    learning_rate_for_step,
    maybe_clip_gradients,
    save_training_checkpoint,
    set_optimizer_lr,
)


DEFAULT_MANIFEST = ROOT / "data" / "corpora.json"
DEFAULT_MODEL_CONFIGS = ROOT / "data" / "model_configs.json"
DEFAULT_OUT_DIR = ROOT / "experiments" / "runs" / "track_a_sandbox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-id", default="toy")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--val-corpus-id", default=None)
    parser.add_argument("--val-corpus", type=Path, default=None)
    parser.add_argument("--allow-unapproved-corpus", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument(
        "--tokenizer-fit-scope",
        choices=("train", "train-val"),
        default="train-val",
    )
    parser.add_argument("--bpe-vocab-size", type=int, default=80)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--model-configs", type=Path, default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--model-config", default="tiny")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--min-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--prompt", default="Muraho")
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("corpus is too small for the requested block size")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def safe_perplexity(loss: float) -> float:
    if loss > 50:
        return float("inf")
    return math.exp(loss)


def encode_prompt(
    tokenizer: AnyTokenizer,
    *,
    prompt: str,
    fallback_text: str,
) -> tuple[str, list[int]]:
    try:
        return prompt, tokenizer.encode(prompt)
    except ValueError:
        fallback = fallback_text[: max(1, min(16, len(fallback_text)))]
        return fallback, tokenizer.encode(fallback)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def interpret_run(status: str) -> str:
    if status == "toy":
        return (
            "Toy sandbox only. A lower loss here proves the loop can learn from "
            "this tiny corpus; it does not prove Kinyarwanda LM quality."
        )
    if status == "approved":
        return (
            "Approved-corpus baseline run. Lower validation loss/perplexity is "
            "a training signal, but it does not prove useful Kinyarwanda "
            "generation without held-out evaluation and fluent-speaker review."
        )
    return (
        "Unapproved/debug corpus run. Use only for local debugging; do not use "
        "the result as model-quality evidence."
    )


def load_train_val_texts(args: argparse.Namespace):
    train_corpus, train_text = load_corpus_text(
        manifest_path=args.manifest,
        corpus_id=args.corpus_id,
        corpus_path=args.corpus,
        allow_unapproved=args.allow_unapproved_corpus,
    )
    if args.val_corpus_id is None and args.val_corpus is None:
        return train_corpus, train_text, None, None

    val_corpus_id = args.val_corpus_id or args.corpus_id
    val_corpus, val_text = load_corpus_text(
        manifest_path=args.manifest,
        corpus_id=val_corpus_id,
        corpus_path=args.val_corpus,
        allow_unapproved=args.allow_unapproved_corpus,
    )
    return train_corpus, train_text, val_corpus, val_text


def split_or_encode_explicit_val(
    *,
    tokenizer: AnyTokenizer,
    train_text: str,
    val_text: str | None,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_token_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    if len(train_token_ids) <= block_size + 1:
        raise ValueError(
            "training corpus is too small for the requested block size: "
            f"{len(train_token_ids)} tokens for block_size={block_size}."
        )

    if val_text is not None:
        val_token_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        if len(val_token_ids) <= block_size:
            raise ValueError(
                "validation corpus is too small for the requested block size: "
                f"{len(val_token_ids)} tokens for block_size={block_size}."
            )
        return train_token_ids, val_token_ids

    split_idx = max(block_size + 1, int(0.9 * len(train_token_ids)))
    split_idx = min(split_idx, len(train_token_ids) - 1)
    return train_token_ids[:split_idx], train_token_ids[split_idx - block_size :]


@torch.no_grad()
def estimate_loss(
    model: TinyTransformerLM,
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int = 5,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(
            data,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    corpus, text, val_corpus, val_text = load_train_val_texts(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (
        load_checkpoint(args.resume_checkpoint, device)
        if args.resume_checkpoint
        else None
    )
    if checkpoint:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        config = TinyTransformerConfig(**checkpoint["config"])
    else:
        tokenizer_text = (
            text
            if args.tokenizer_fit_scope == "train" or val_text is None
            else text + "\n" + val_text
        )
        tokenizer = train_tokenizer(
            args.tokenizer,
            tokenizer_text,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_min_frequency=args.bpe_min_frequency,
        )
        config = build_model_config(
            presets_path=args.model_configs,
            preset_name=args.model_config,
            vocab_size=tokenizer.vocab_size,
            overrides={
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
            },
        )
    train_data, val_data = split_or_encode_explicit_val(
        tokenizer=tokenizer,
        train_text=text,
        val_text=val_text,
        block_size=config.block_size,
    )
    total_characters = len(text) + (len(val_text) if val_text else 0)
    total_tokens = len(train_data) + len(val_data)

    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_time = time.time()
    losses: list[dict[str, float | int]] = []
    initial_loss = estimate_loss(
        model,
        val_data,
        batch_size=args.batch_size,
        block_size=config.block_size,
        device=device,
        eval_iters=args.eval_iters,
    )
    initial_perplexity = safe_perplexity(initial_loss)

    for step in range(1, args.max_steps + 1):
        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            block_size=config.block_size,
            device=device,
        )
        lr = learning_rate_for_step(
            step=step,
            max_steps=args.max_steps,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            schedule=args.lr_schedule,
        )
        set_optimizer_lr(optimizer, lr)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = maybe_clip_gradients(model, max_norm=args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size=args.batch_size,
                block_size=config.block_size,
                device=device,
                eval_iters=args.eval_iters,
            )
            losses.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                }
            )
            print(
                f"step {step:04d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f}"
            )
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save_training_checkpoint(
                path=args.out_dir / f"checkpoint_step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                summary={"step": step, "loss": float(loss.item())},
            )

    prompt, prompt_token_ids = encode_prompt(
        tokenizer,
        prompt=args.prompt,
        fallback_text=text,
    )
    prompt_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, max_new_tokens=args.sample_tokens)
    sample = tokenizer.decode(generated[0].tolist())
    final_loss = float(losses[-1]["val_loss"] if losses else initial_loss)
    final_perplexity = safe_perplexity(final_loss)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoint.pt" if args.save_checkpoint else None
    tokenizer_payload = tokenizer.to_dict()
    previous_tokenizer_summary = (
        checkpoint.get("summary", {}).get("tokenizer", {}) if checkpoint else {}
    )
    summary = {
        "args": serializable_args(args),
        "corpus": corpus.to_dict(),
        "validation_corpus": val_corpus.to_dict() if val_corpus else None,
        "corpus_path": str(corpus.path),
        "device": str(device),
        "seed": args.seed,
        "tokenizer": {
            "type": tokenizer_payload["type"],
            "vocab_size": tokenizer.vocab_size,
            "num_merges": len(tokenizer_payload["merges"]),
            "fit_scope": (
                previous_tokenizer_summary.get("fit_scope")
                if checkpoint
                else args.tokenizer_fit_scope
            ),
            "bpe_min_frequency": (
                args.bpe_min_frequency if tokenizer_payload["type"] == "bpe" else None
            ),
        },
        "vocab_size": tokenizer.vocab_size,
        "num_characters": total_characters,
        "num_tokens": total_tokens,
        "tokens_per_character": round(total_tokens / total_characters, 4),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "config": asdict(config),
        "model_config": args.model_config,
        "max_steps": args.max_steps,
        "eval_iters": args.eval_iters,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "checkpoint_interval": args.checkpoint_interval,
        "initial_val_loss": float(initial_loss),
        "final_val_loss": final_loss,
        "initial_val_perplexity": initial_perplexity,
        "final_val_perplexity": final_perplexity,
        "losses": losses,
        "prompt": prompt,
        "sample": sample,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "interpretation": interpret_run(corpus.status),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "run_report.md").write_text(
        render_run_report(summary),
        encoding="utf-8",
    )
    (args.out_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(
        json.dumps(list(tokenizer.to_dict()["vocab"]), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if checkpoint_path:
        save_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            summary=summary,
        )

    print(f"initial_val_loss={initial_loss:.4f}")
    print(f"final_val_loss={final_loss:.4f}")
    print(f"initial_val_perplexity={initial_perplexity:.4f}")
    print(f"final_val_perplexity={final_perplexity:.4f}")
    if checkpoint_path:
        print(f"checkpoint={checkpoint_path}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
