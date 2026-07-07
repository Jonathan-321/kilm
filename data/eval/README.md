# Kinyarwanda Conversation Benchmark

Use this folder for held-out tasks that are never trained on.

Recommended benchmark rows:

```json
{"id": "conv-001", "category": "conversation", "prompt": "...", "reference": "..."}
```

or:

```json
{"id": "qa-001", "category": "reading_qa", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Run a model and create a fluent-speaker review sheet:

```bash
python scripts/run_conversation_benchmark.py \
  --model checkpoints/kilm-llama-100m/checkpoint-50000 \
  --benchmark data/eval/kinyarwanda_conversation_benchmark.jsonl
```
