"""Create data/model cards and a sample review sheet from a run."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.reporting import load_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_summary(args.run)
    out_dir = args.out_dir or args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "data_card.md").write_text(render_data_card(summary), encoding="utf-8")
    (out_dir / "model_card.md").write_text(render_model_card(summary), encoding="utf-8")
    write_sample_review(out_dir / "sample_review.tsv", summary)

    print(f"wrote {out_dir / 'data_card.md'}")
    print(f"wrote {out_dir / 'model_card.md'}")
    print(f"wrote {out_dir / 'sample_review.tsv'}")
    return 0


def render_data_card(summary: dict[str, object]) -> str:
    corpus = summary["corpus"]
    val_corpus = summary.get("validation_corpus")
    lines = [
        "# Data Card",
        "",
        "## Training Corpus",
        "",
        f"- ID: `{corpus['id']}`",
        f"- Status: `{corpus['status']}`",
        f"- Source: {corpus.get('source') or 'n/a'}",
        f"- License: `{corpus.get('license') or 'n/a'}`",
        f"- Path: `{corpus['path']}`",
        f"- Train tokens: `{summary['train_tokens']}`",
        "",
    ]
    if val_corpus:
        lines.extend(
            [
                "## Validation Corpus",
                "",
                f"- ID: `{val_corpus['id']}`",
                f"- Status: `{val_corpus['status']}`",
                f"- Path: `{val_corpus['path']}`",
                f"- Validation tokens: `{summary['val_tokens']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Known Limits",
            "",
            "- This card is generated from local run metadata.",
            "- Approval status comes from the manifest, not from legal advice.",
            "- Speaker/domain quality review is still required before claims.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_model_card(summary: dict[str, object]) -> str:
    tokenizer = summary["tokenizer"]
    config = summary["config"]
    lines = [
        "# Model Card",
        "",
        "## Run",
        "",
        f"- Corpus: `{summary['corpus']['id']}`",
        f"- Validation corpus: `{(summary.get('validation_corpus') or {}).get('id', 'internal split')}`",
        f"- Tokenizer: `{tokenizer['type']}`",
        f"- Vocab size: `{tokenizer['vocab_size']}`",
        f"- Model config: `{summary.get('model_config', 'custom')}`",
        f"- Layers: `{config['n_layer']}`",
        f"- Heads: `{config['n_head']}`",
        f"- Embedding dim: `{config['n_embd']}`",
        f"- Block size: `{config['block_size']}`",
        "",
        "## Metrics",
        "",
        f"- Initial validation loss: `{summary['initial_val_loss']:.4f}`",
        f"- Final validation loss: `{summary['final_val_loss']:.4f}`",
        f"- Initial validation perplexity: `{summary['initial_val_perplexity']:.4f}`",
        f"- Final validation perplexity: `{summary['final_val_perplexity']:.4f}`",
        "",
        "## Intended Use",
        "",
        "Research and class-project baseline only.",
        "",
        "## Limitations",
        "",
        "- Do not use generated text as authoritative Kinyarwanda.",
        "- Human sample review is required before usefulness claims.",
        "- More data and longer training are required before scaling claims.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_sample_review(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt",
                "sample",
                "reviewer",
                "fluency_1_5",
                "meaningfulness_1_5",
                "grammar_notes",
                "safety_or_bias_notes",
                "decision",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt": summary["prompt"],
                "sample": str(summary["sample"]).replace("\n", "\\n"),
                "reviewer": "",
                "fluency_1_5": "",
                "meaningfulness_1_5": "",
                "grammar_notes": "",
                "safety_or_bias_notes": "",
                "decision": "needs-review",
            }
        )


if __name__ == "__main__":
    sys.exit(main())
