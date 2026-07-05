"""Model config preset loading."""

from __future__ import annotations

import json
from pathlib import Path

from kilm.tiny_transformer import TinyTransformerConfig


def load_model_presets(path: Path) -> dict[str, dict[str, int | float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_config(
    *,
    presets_path: Path,
    preset_name: str,
    vocab_size: int,
    overrides: dict[str, int | float | None],
) -> TinyTransformerConfig:
    presets = load_model_presets(presets_path)
    if preset_name not in presets:
        known = ", ".join(sorted(presets))
        raise ValueError(f"unknown model preset {preset_name!r}; known: {known}")

    values = dict(presets[preset_name])
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TinyTransformerConfig(vocab_size=vocab_size, **values)
