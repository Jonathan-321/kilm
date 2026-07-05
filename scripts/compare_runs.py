"""Compare KILM run summary artifacts."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kilm.reporting import load_summary, render_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "analysis" / "run_comparison.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [(run.stem, load_summary(run)) for run in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_comparison_report(summaries), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
