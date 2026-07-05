"""Training utilities for KILM."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import torch

from kilm.tiny_transformer import TinyTransformerConfig, TinyTransformerLM
from kilm.tokenizers import AnyTokenizer


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not _mps_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"unknown device: {requested}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def learning_rate_for_step(
    *,
    step: int,
    max_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule: {schedule}")

    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_clip_gradients(
    model: TinyTransformerLM,
    *,
    max_norm: float | None,
) -> float | None:
    if max_norm is None or max_norm <= 0:
        return None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(norm.item())


def save_training_checkpoint(
    *,
    path: Path,
    model: TinyTransformerLM,
    optimizer: torch.optim.Optimizer,
    config: TinyTransformerConfig,
    tokenizer: AnyTokenizer,
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "summary": summary,
        },
        path,
    )
