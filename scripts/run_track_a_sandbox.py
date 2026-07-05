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

from kilm.bpe_tokenizer import BpeTokenizer
from kilm.char_tokenizer import CharTokenizer
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM


DEFAULT_CORPUS = ROOT / "data" / "toy_corpus.txt"
DEFAULT_OUT_DIR = ROOT / "experiments" / "runs" / "track_a_sandbox"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--bpe-vocab-size", type=int, default=80)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--prompt", default="Muraho")
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


def build_tokenizer(args: argparse.Namespace, text: str) -> CharTokenizer | BpeTokenizer:
    if args.tokenizer == "char":
        return CharTokenizer.train(text)
    return BpeTokenizer.train(
        text,
        vocab_size=args.bpe_vocab_size,
        min_frequency=args.bpe_min_frequency,
    )


def safe_perplexity(loss: float) -> float:
    if loss > 50:
        return float("inf")
    return math.exp(loss)


def encode_prompt(
    tokenizer: CharTokenizer | BpeTokenizer,
    *,
    prompt: str,
    fallback_text: str,
) -> tuple[str, list[int]]:
    try:
        return prompt, tokenizer.encode(prompt)
    except ValueError:
        fallback = fallback_text[: max(1, min(16, len(fallback_text)))]
        return fallback, tokenizer.encode(fallback)


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
        x, y = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = build_tokenizer(args, text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(token_ids) <= args.block_size + 1:
        raise ValueError(
            "encoded corpus is too small for the requested block size: "
            f"{len(token_ids)} tokens for block_size={args.block_size}. "
            "Use a larger corpus, a smaller BPE vocab, or a smaller block size."
        )

    split_idx = max(args.block_size + 1, int(0.9 * len(token_ids)))
    split_idx = min(split_idx, len(token_ids) - 1)
    train_data = token_ids[:split_idx]
    val_data = token_ids[split_idx - args.block_size :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TinyTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_time = time.time()
    losses: list[dict[str, float | int]] = []
    initial_loss = estimate_loss(
        model,
        val_data,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
    )
    initial_perplexity = safe_perplexity(initial_loss)

    for step in range(1, args.max_steps + 1):
        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
            )
            losses.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                }
            )
            print(f"step {step:04d} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

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
    summary = {
        "corpus": str(args.corpus),
        "device": str(device),
        "seed": args.seed,
        "tokenizer": {
            "type": args.tokenizer,
            "vocab_size": tokenizer.vocab_size,
            "num_merges": len(tokenizer.to_dict()["merges"]),
            "bpe_min_frequency": (
                args.bpe_min_frequency if args.tokenizer == "bpe" else None
            ),
        },
        "vocab_size": tokenizer.vocab_size,
        "num_characters": len(text),
        "num_tokens": len(token_ids),
        "tokens_per_character": round(len(token_ids) / len(text), 4),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "config": asdict(config),
        "max_steps": args.max_steps,
        "initial_val_loss": float(initial_loss),
        "final_val_loss": final_loss,
        "initial_val_perplexity": initial_perplexity,
        "final_val_perplexity": final_perplexity,
        "losses": losses,
        "prompt": prompt,
        "sample": sample,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "interpretation": (
            "Toy sandbox only. A lower loss here proves the loop can learn from "
            "this tiny corpus; it does not prove Kinyarwanda LM quality."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
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

    print(f"initial_val_loss={initial_loss:.4f}")
    print(f"final_val_loss={final_loss:.4f}")
    print(f"initial_val_perplexity={initial_perplexity:.4f}")
    print(f"final_val_perplexity={final_perplexity:.4f}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
