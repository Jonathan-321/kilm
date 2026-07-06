"""Print the latest KILM training status from a Trainer log file."""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import re
import sys


DEFAULT_LOG = Path("logs") / "cuda_training.log"
DICT_RE = re.compile(r"\{[^{}]+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--logging-steps", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2

    status = read_status(args.log, logging_steps=args.logging_steps)
    if not status["loss_records"] and status["latest_eval_loss"] is None:
        print(f"No Trainer metrics found yet in {args.log}")
        return 1

    print(f"log={args.log}")
    print(f"estimated_step={status['estimated_step']}")
    print(f"latest_loss={status['latest_loss']}")
    print(f"latest_grad_norm={status['latest_grad_norm']}")
    print(f"latest_learning_rate={status['latest_learning_rate']}")
    print(f"latest_epoch={status['latest_epoch']}")
    print(f"latest_eval_loss={status['latest_eval_loss']}")
    print(f"latest_eval_runtime={status['latest_eval_runtime']}")
    return 0


def read_status(log_path: Path, *, logging_steps: int) -> dict[str, object]:
    loss_records: list[dict[str, object]] = []
    eval_records: list[dict[str, object]] = []

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for match in DICT_RE.findall(line):
            try:
                payload = ast.literal_eval(match)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if "loss" in payload:
                loss_records.append(payload)
            if "eval_loss" in payload:
                eval_records.append(payload)

    latest_loss = loss_records[-1] if loss_records else {}
    latest_eval = eval_records[-1] if eval_records else {}
    return {
        "loss_records": loss_records,
        "estimated_step": len(loss_records) * logging_steps,
        "latest_loss": latest_loss.get("loss"),
        "latest_grad_norm": latest_loss.get("grad_norm"),
        "latest_learning_rate": latest_loss.get("learning_rate"),
        "latest_epoch": latest_loss.get("epoch") or latest_eval.get("epoch"),
        "latest_eval_loss": latest_eval.get("eval_loss"),
        "latest_eval_runtime": latest_eval.get("eval_runtime"),
    }


if __name__ == "__main__":
    sys.exit(main())
