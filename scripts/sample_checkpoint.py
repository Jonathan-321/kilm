"""Generate text from a saved KILM checkpoint."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from kilm.checkpointing import load_checkpoint
from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from kilm.tokenizers import tokenizer_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Muraho")
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    config = TinyTransformerConfig(**checkpoint["config"])
    model = TinyTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        prompt_ids = tokenizer.encode(args.prompt)
        prompt = args.prompt
    except ValueError:
        prompt = checkpoint.get("summary", {}).get("prompt", "")
        if not prompt:
            raise
        prompt_ids = tokenizer.encode(prompt)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        ids,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
    )
    print(tokenizer.decode(generated[0].tolist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
