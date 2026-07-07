# Kinyarwanda SFT Data

Put real user/assistant conversation data here before supervised fine-tuning.
Do not train on synthetic placeholder text as if it were ground truth.

Accepted input schemas:

```json
{"user": "Muraho, wamfasha kwiga Ikinyarwanda?", "assistant": "Yego, reka duhere ku nteruro yoroshye."}
```

or:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Prepare train/validation splits:

```bash
python scripts/prepare_sft_conversations.py \
  --input data/sft/raw_conversations.jsonl \
  --out-dir data/sft/processed \
  --validation-fraction 0.1
```
