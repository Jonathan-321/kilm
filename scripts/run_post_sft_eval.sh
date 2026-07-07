#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-checkpoints/sft/qwen2.5-7b-kinyarwanda-qlora}"
MERGED_MODEL="${MERGED_MODEL:-checkpoints/sft/qwen2.5-7b-kinyarwanda-merged}"
BENCHMARK="${BENCHMARK:-data/eval/kinyarwanda_conversation_benchmark.jsonl}"
OUT_DIR="${OUT_DIR:-eval_results}"
LIMIT="${LIMIT:-20}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter not found: $ADAPTER" >&2
  exit 1
fi

if [[ ! -f "$BENCHMARK" ]]; then
  echo "Benchmark not found: $BENCHMARK" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output-dir "$MERGED_MODEL"

python scripts/run_conversation_benchmark.py \
  --model "$MERGED_MODEL" \
  --benchmark "$BENCHMARK" \
  --out-dir "$OUT_DIR" \
  --limit "$LIMIT"

latest_review="$(ls -t "$OUT_DIR"/conversation_benchmark_review_*.tsv | head -1)"
cp "$latest_review" "$OUT_DIR/sft_benchmark_review.tsv"
echo "review_sheet=$OUT_DIR/sft_benchmark_review.tsv"
