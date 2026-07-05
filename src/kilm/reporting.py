"""Markdown report helpers for KILM experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict[str, object]:
    summary_path = path / "summary.json" if path.is_dir() else path
    return json.loads(summary_path.read_text(encoding="utf-8"))


def render_run_report(summary: dict[str, object]) -> str:
    corpus = summary["corpus"]
    tokenizer = summary["tokenizer"]
    config = summary["config"]

    lines = [
        "# KILM Run Report",
        "",
        "## Result",
        "",
        f"- Final validation loss: `{_fmt(summary['final_val_loss'])}`",
        f"- Final validation perplexity: `{_fmt(summary['final_val_perplexity'])}`",
        f"- Initial validation loss: `{_fmt(summary['initial_val_loss'])}`",
        f"- Initial validation perplexity: `{_fmt(summary['initial_val_perplexity'])}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']}`",
        "",
        "## Corpus",
        "",
        f"- ID: `{corpus['id']}`",
        f"- Status: `{corpus['status']}`",
        f"- Path: `{corpus['path']}`",
        f"- Description: {corpus.get('description') or 'n/a'}",
    ]
    if summary.get("validation_corpus"):
        val_corpus = summary["validation_corpus"]
        lines.extend(
            [
                "",
                "## Validation Corpus",
                "",
                f"- ID: `{val_corpus['id']}`",
                f"- Status: `{val_corpus['status']}`",
                f"- Path: `{val_corpus['path']}`",
                f"- Description: {val_corpus.get('description') or 'n/a'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Tokenizer",
            "",
            f"- Type: `{tokenizer['type']}`",
            f"- Fit scope: `{tokenizer.get('fit_scope', 'n/a')}`",
            f"- Vocab size: `{tokenizer['vocab_size']}`",
            f"- Merges: `{tokenizer['num_merges']}`",
            f"- Tokens: `{summary['num_tokens']}`",
            f"- Tokens/character: `{summary['tokens_per_character']}`",
            "",
            "## Model",
            "",
            f"- Layers: `{config['n_layer']}`",
            f"- Heads: `{config['n_head']}`",
            f"- Embedding dim: `{config['n_embd']}`",
            f"- Block size: `{config['block_size']}`",
            f"- Train tokens: `{summary['train_tokens']}`",
            f"- Validation tokens: `{summary['val_tokens']}`",
            "",
            "## Training",
            "",
            f"- Max steps: `{summary.get('max_steps', 'n/a')}`",
            f"- Evaluation batches: `{summary.get('eval_iters', 'n/a')}`",
            f"- Learning-rate schedule: `{summary.get('lr_schedule', 'n/a')}`",
            f"- Warmup steps: `{summary.get('warmup_steps', 'n/a')}`",
            f"- Gradient clipping: `{summary.get('grad_clip', 'n/a')}`",
            f"- Checkpoint interval: `{summary.get('checkpoint_interval', 'n/a')}`",
            "",
            "## Sample",
            "",
            f"Prompt: `{summary['prompt']}`",
            "",
            "```text",
            str(summary["sample"]).rstrip(),
            "```",
            "",
            "## Interpretation",
            "",
            str(summary["interpretation"]),
        ]
    )
    if summary.get("losses"):
        lines.extend(
            [
                "",
                "## Loss Trace",
                "",
                "| step | train loss | val loss | lr | grad norm |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["losses"]:
            lines.append(
                "| {step} | {train_loss} | {val_loss} | {lr} | {grad_norm} |".format(
                    step=row["step"],
                    train_loss=_fmt(row["train_loss"]),
                    val_loss=_fmt(row["val_loss"]),
                    lr=_fmt(row.get("learning_rate", "n/a")),
                    grad_norm=_fmt(row.get("grad_norm", "n/a")),
                )
            )
    if summary.get("checkpoint"):
        lines.extend(["", "## Checkpoint", "", f"`{summary['checkpoint']}`"])
    if summary.get("resumed_from"):
        lines.extend(["", "## Resumed From", "", f"`{summary['resumed_from']}`"])
    return "\n".join(lines).rstrip() + "\n"


def render_comparison_report(
    summaries: list[tuple[str, dict[str, object]]],
) -> str:
    lines = [
        "# KILM Run Comparison",
        "",
        "| run | corpus | status | tokenizer | vocab | tokens | final loss | "
        "final ppl | checkpoint |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, summary in summaries:
        corpus = summary["corpus"]
        tokenizer = summary["tokenizer"]
        checkpoint = "yes" if summary.get("checkpoint") else "no"
        lines.append(
            "| {label} | {corpus_id} | {status} | {tokenizer_type} | "
            "{vocab} | {tokens} | {loss} | {ppl} | {checkpoint} |".format(
                label=label,
                corpus_id=corpus["id"],
                status=corpus["status"],
                tokenizer_type=tokenizer["type"],
                vocab=tokenizer["vocab_size"],
                tokens=summary["num_tokens"],
                loss=_fmt(summary["final_val_loss"]),
                ppl=_fmt(summary["final_val_perplexity"]),
                checkpoint=checkpoint,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
