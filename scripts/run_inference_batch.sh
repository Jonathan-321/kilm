#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-checkpoints/kilm-llama-100m/checkpoint-50000}"
OUT_DIR="${OUT_DIR:-logs/inference}"
PROMPT_FILE="${1:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
SEED="${SEED:-1337}"
GREEDY="${GREEDY:-0}"

mkdir -p "$OUT_DIR"

if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$OUT_DIR/default_prompts.txt"
  if [[ ! -s "$PROMPT_FILE" ]]; then
    cat > "$PROMPT_FILE" <<'PROMPTS'
Muraho
Mu Rwanda
Abaturage ba Kigali
Uburezi mu Rwanda
Ubuzima mu Rwanda
Perezida Kagame yavuze ko
Leta y'u Rwanda yatangaje ko
Abahinzi borozi mu Rwanda
Ikoranabuhanga mu mashuri
Urubyiruko rw'u Rwanda
Umuryango nyarwanda
Iterambere ry'icyaro
Ubukungu bw'u Rwanda
Amakuru y'u Rwanda uyu munsi
Siporo mu Rwanda
PROMPTS
  fi
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT" >&2
  echo "Set CHECKPOINT=/path/to/checkpoint if you want a different one." >&2
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODE="sampled"
GEN_ARGS=(
  python scripts/generate_kilm.py
  --checkpoint "$CHECKPOINT"
  --prompt-file "$PROMPT_FILE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --num-return-sequences "$NUM_RETURN_SEQUENCES"
  --seed "$SEED"
)

if [[ "$GREEDY" == "1" ]]; then
  MODE="greedy"
  GEN_ARGS+=(--greedy)
else
  GEN_ARGS+=(
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
  )
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_FILE:-$OUT_DIR/kilm_${MODE}_${STAMP}.txt}"

{
  echo "checkpoint=$CHECKPOINT"
  echo "prompt_file=$PROMPT_FILE"
  echo "output=$OUT_FILE"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "num_return_sequences=$NUM_RETURN_SEQUENCES"
  if [[ "$GREEDY" == "1" ]]; then
    echo "decode=greedy"
  else
    echo "decode=sampled temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P repetition_penalty=$REPETITION_PENALTY"
  fi
  echo
  "${GEN_ARGS[@]}"
} | tee "$OUT_FILE"

echo
echo "Saved outputs to $OUT_FILE"
