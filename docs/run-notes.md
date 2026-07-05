# Track A Sandbox Run Notes

## 2026-07-05 Pipeline Artifact Run

Commands:

```bash
python3 scripts/analyze_tokenizers.py \
  --bpe-vocab-size 64 \
  --out-dir experiments/analysis/tokenizers_smoke

python3 scripts/run_track_a_sandbox.py \
  --tokenizer bpe \
  --bpe-vocab-size 64 \
  --max-steps 40 \
  --out-dir experiments/runs/bpe_checkpoint_smoke

python3 scripts/sample_checkpoint.py \
  experiments/runs/bpe_checkpoint_smoke/checkpoint.pt \
  --prompt Muraho \
  --sample-tokens 80
```

Result:

```text
tests: 11 passed
tokenizer analysis: char 558 tokens, BPE 386 tokens
BPE final_val_loss=1.8230
BPE final_val_perplexity=6.1904
checkpoint saved and loadable
```

Interpretation:

KILM now produces reusable pipeline artifacts, not just console output. Each run
is attached to a corpus manifest record, writes tokenizer metadata, can save a
checkpoint, and can generate again from that checkpoint later.

The result is still toy-only. The next serious step is to put a small approved
Kinyarwanda corpus into the manifest and rerun the same analysis/training flow.

## 2026-07-05 Tiny BPE LM Run

Command:

```bash
python3 scripts/run_track_a_sandbox.py \
  --tokenizer bpe \
  --bpe-vocab-size 64 \
  --max-steps 40 \
  --out-dir experiments/runs/bpe_smoke
```

Result:

```text
initial_val_loss=4.3828
final_val_loss=1.8230
initial_val_perplexity=80.0588
final_val_perplexity=6.1904
vocab_size=64
num_merges=24
num_tokens=386
tokens_per_character=0.6918
```

Interpretation:

The BPE path now runs through the same full loop as the character baseline:

```text
toy text -> BPE tokenizer -> token IDs -> tiny Transformer -> loss/perplexity -> sample
```

Compared with the character baseline, BPE compresses the toy corpus from 558
tokens to 386 tokens. Validation loss still drops on the toy run, so the model
loop works with merged tokens too.

This still does not prove model quality. The corpus is tiny and explicitly not
approved training data. The next real gate is an approved small corpus, then the
same char-vs-BPE comparison should be rerun on that text.

## 2026-07-04 Tiny Character LM Run

Command:

```bash
python3 scripts/run_track_a_sandbox.py --max-steps 40
```

Result:

```text
initial_val_loss=3.9356
final_val_loss=1.7618
vocab_size=40
num_tokens=558
```

Interpretation:

The toy model learned signal from the tiny toy corpus: validation loss dropped
quickly. That means the end-to-end loop is wired correctly:

```text
toy text -> tokenizer -> token IDs -> tiny Transformer -> loss -> sample
```

The sample is not useful Kinyarwanda. It contains fragments that resemble the
toy corpus, but it is mostly incoherent. That is expected because the corpus is
tiny, the tokenizer is character-level, and the model is trained for only a few
steps.

What this proves:

- the local training loop works,
- the model can overfit/learn from text,
- the team has a safe guide for the full Track A shape.

What this does not prove:

- that we have enough data,
- that the tokenizer is good,
- that generation quality will be useful,
- that Track B can be dropped.

Next gate:

Rerun the char-vs-BPE comparison and tiny LM loop on a small approved corpus.
