# KILM SFT Run Report

## Goal

The from-scratch KILM model proved the pretraining pipeline, but its samples
are not usable as a tutor. The SFT path adapts an existing multilingual
instruction model to Kinyarwanda user/assistant behavior.

## Base Model

- Primary: `Qwen/Qwen2.5-7B-Instruct`
- Fallback: `mistralai/Mistral-7B-Instruct-v0.2`
- Method: QLoRA, 4-bit NF4 quantization
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- LoRA rank: `16`
- LoRA alpha: `32`

## Data

Raw conversational bootstrap data is written to:

```text
data/sft/raw_conversations.jsonl
```

Processed SFT splits are written to:

```text
data/sft/processed/train.jsonl
data/sft/processed/validation.jsonl
```

The bootstrap script searches open Hugging Face datasets for Kinyarwanda
translation pairs and converts them into instruction-style user/assistant
examples. This is a starting point, not a substitute for fluent-speaker
conversation data.

## Launch Command

```bash
tmux new-session -d -s kilm_sft "
  cd ~/kilm &&
  source .venv/bin/activate &&
  mkdir -p logs &&
  PYTHONUNBUFFERED=1 python scripts/train_sft_qlora.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --fallback-model mistralai/Mistral-7B-Instruct-v0.2 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --max-seq-length 2048 \
    --num-train-epochs 3 \
    --target-modules q_proj k_proj v_proj o_proj \
    2>&1 | tee logs/sft_training.log
"
```

## Monitor

```bash
tmux attach -t kilm_sft
tail -f logs/sft_training.log
watch -n 5 nvidia-smi
```

## Post-SFT Evaluation

After training finishes:

```bash
bash scripts/run_post_sft_eval.sh
```

The review sheet is written to:

```text
eval_results/sft_benchmark_review.tsv
```
