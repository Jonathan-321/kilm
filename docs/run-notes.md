# Track A Sandbox Run Notes

## 2026-07-06 Rebuilt 109M KILM Baseline

Commands:

```bash
.venv/bin/python scripts/aggregate_corpus.py
.venv/bin/python scripts/train_sentencepiece.py
.venv/bin/python scripts/tokenize_corpus.py --num-proc 4

PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_kilm.py \
  --max-steps 2000 \
  --gradient-accumulation-steps 1 \
  --logging-steps 100 \
  --disable-tqdm

.venv/bin/python scripts/write_final_report.py
```

Result:

```text
corpus_words=22519811
tokenized_train_tokens=33985536
tokenized_validation_tokens=711680
total_tokenized_tokens=34697216
tokenizer=SentencePiece BPE 32000 byte fallback
model_parameters=109529856
completed_steps=2000
train_loss=6.818963836669922
train_perplexity=915.036391549543
validation_loss=5.864002704620361
validation_perplexity=352.13080251279723
checkpoint=checkpoints/kilm-llama-100m/checkpoint-2000
```

Interpretation:

The rebuilt pipeline works end to end on the large corpus: aggregation,
SentencePiece tokenizer training, Arrow tokenization, 109M-parameter
from-scratch training, checkpointing, validation, sampling, and final reporting
all completed. This is a real baseline, not a toy loop.

The samples are no longer pure character garbage, but they are still repetitive
and not semantically reliable. The current model decision is
`needs-more-training-and-fluent-review`. A real next run should resume from
`checkpoint-2000` on stronger CUDA hardware and continue toward the default
50000-step target.

## 2026-07-05 Approved MT Baseline GPU MPS 10k Continuation

Commands:

```bash
python3 scripts/fetch_approved_corpus.py \
  --source digital-umuganda-mt-rw \
  --min-line-chars 5

python3 scripts/prepare_corpus.py \
  --corpus-id digital-umuganda-mt-rw \
  --out-dir data/processed/digital_umuganda_mt_full \
  --val-fraction 0.02 \
  --min-line-chars 5

python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/digital_umuganda_mt_full/corpora.json \
  --corpus-id digital-umuganda-mt-rw-train \
  --val-corpus-id digital-umuganda-mt-rw-val \
  --tokenizer bpe \
  --tokenizer-fit-scope train-val \
  --bpe-vocab-size 512 \
  --model-config baseline_gpu \
  --max-steps 2000 \
  --eval-interval 200 \
  --eval-iters 5 \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --min-learning-rate 0.00005 \
  --lr-schedule cosine \
  --warmup-steps 100 \
  --grad-clip 1.0 \
  --checkpoint-interval 500 \
  --sample-interval 250 \
  --sample-temperature 0.7 \
  --sample-top-k 40 \
  --sample-tokens 220 \
  --device mps \
  --out-dir experiments/runs/du_mt_full_baseline_gpu_mps_2k

python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/digital_umuganda_mt_full/corpora.json \
  --corpus-id digital-umuganda-mt-rw-train \
  --val-corpus-id digital-umuganda-mt-rw-val \
  --resume-checkpoint experiments/runs/du_mt_full_baseline_gpu_mps_2k/checkpoint.pt \
  --max-steps 10000 \
  --eval-interval 500 \
  --eval-iters 5 \
  --batch-size 8 \
  --learning-rate 0.00005 \
  --min-learning-rate 0.00005 \
  --lr-schedule constant \
  --grad-clip 1.0 \
  --checkpoint-interval 2000 \
  --sample-interval 1000 \
  --sample-temperature 0.7 \
  --sample-top-k 40 \
  --sample-tokens 220 \
  --device mps \
  --out-dir experiments/runs/du_mt_full_baseline_gpu_mps_continue_10k

python3 scripts/create_review_packet.py \
  experiments/runs/du_mt_full_baseline_gpu_mps_continue_10k \
  --sample-decision needs-linguistic-review
```

Result:

```text
prepared_lines=44527
train_lines=43636
val_lines=891
replacement_chars=0
BPE vocab=512
train_tokens=764213
val_tokens=15420
model=baseline_gpu
block_size=256
2k perplexity=599.4842 -> 42.1314
10k continuation perplexity=43.7940 -> 21.0469
```

Interpretation:

This is the first run where more approved data and a larger model/context
clearly beat the TTS-only path. The final sample is still grammatically and
semantically unreliable, but it is no longer pure garbage. It is marked
`needs-linguistic-review`, not as a usable learner-facing result.

Important implementation note: the MT source files are cp1252-style text, not
UTF-8. The importer now falls back to cp1252 decoding so the prepared text does
not contain replacement characters.

## 2026-07-05 Approved TTS 10k Continuation

Command:

```bash
python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/digital_umuganda_tts_full/corpora.json \
  --corpus-id digital-umuganda-tts-rw-train \
  --val-corpus-id digital-umuganda-tts-rw-val \
  --resume-checkpoint experiments/runs/du_tts_full_small_mps_baseline/checkpoint.pt \
  --max-steps 10000 \
  --eval-interval 500 \
  --eval-iters 10 \
  --batch-size 32 \
  --learning-rate 0.00008 \
  --min-learning-rate 0.00008 \
  --lr-schedule constant \
  --grad-clip 1.0 \
  --checkpoint-interval 2000 \
  --sample-interval 1000 \
  --sample-temperature 0.7 \
  --sample-top-k 40 \
  --sample-tokens 220 \
  --device mps \
  --out-dir experiments/runs/du_tts_full_small_mps_continue_10k

python3 scripts/create_review_packet.py \
  experiments/runs/du_tts_full_small_mps_continue_10k \
  --sample-decision failed-smoke
```

Result:

```text
resumed_from=experiments/runs/du_tts_full_small_mps_baseline/checkpoint.pt
initial_val_loss=4.9357
final_val_loss=4.0865
initial_val_perplexity=139.1711
final_val_perplexity=59.5324
elapsed_seconds=593.329
sample_snapshots=10
checkpoints=2000,4000,6000,8000,10000
```

Interpretation:

Longer training kept working: validation perplexity dropped from about 139 to
about 60 after 10,000 continuation steps. The generated samples are still not
usable Kinyarwanda, so the current sample review decision is `failed-smoke`, not
`needs-review`.

The next technical move is not just more of the same forever. We should try a
larger context/model and more approved text, then compare whether sample quality
improves at the same or lower validation loss.

## 2026-07-05 Approved TTS Full Small MPS Baseline

Commands:

```bash
python3 scripts/fetch_approved_corpus.py \
  --source digital-umuganda-tts-rw

python3 scripts/prepare_corpus.py \
  --corpus-id digital-umuganda-tts-rw \
  --out-dir data/processed/digital_umuganda_tts_full \
  --val-fraction 0.1 \
  --min-line-chars 5

python3 scripts/analyze_tokenizers.py \
  --manifest data/processed/digital_umuganda_tts_full/corpora.json \
  --corpus-id digital-umuganda-tts-rw-full \
  --bpe-vocab-size 512 \
  --out-dir experiments/analysis/du_tts_full_tokenizers

python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/digital_umuganda_tts_full/corpora.json \
  --corpus-id digital-umuganda-tts-rw-train \
  --val-corpus-id digital-umuganda-tts-rw-val \
  --tokenizer bpe \
  --tokenizer-fit-scope train-val \
  --bpe-vocab-size 512 \
  --model-config small \
  --max-steps 200 \
  --eval-interval 25 \
  --eval-iters 5 \
  --batch-size 32 \
  --learning-rate 0.0008 \
  --min-learning-rate 0.00008 \
  --lr-schedule cosine \
  --warmup-steps 10 \
  --grad-clip 1.0 \
  --checkpoint-interval 50 \
  --device mps \
  --out-dir experiments/runs/du_tts_full_small_mps_baseline

python3 scripts/create_review_packet.py \
  experiments/runs/du_tts_full_small_mps_baseline
```

Result:

```text
prepared_lines=3922
train_lines=3530
val_lines=392
char tokens=340384
BPE tokens=134090
BPE tokens/word=2.8540
train_tokens=120605
val_tokens=13481
initial_val_loss=6.4065
final_val_loss=4.9201
initial_val_perplexity=605.7486
final_val_perplexity=137.0228
device=mps
interval checkpoints written at steps 50, 100, 150, and 200
```

Interpretation:

This is the first larger approved-data baseline. It uses the full local
Digital Umuganda TTS sentence text, the `small` config, Apple MPS, cosine LR
decay, gradient clipping, real held-out validation, and review-card outputs.

The sample is still not ready for learner use, but it is much less toy-like
than the 20-step sanity run. The next useful scale step is longer training and
speaker review, not another rewrite of the loop.

## 2026-07-05 Approved TTS 1k Baseline

Commands:

```bash
python3 scripts/fetch_approved_corpus.py \
  --source digital-umuganda-tts-rw \
  --limit 1000

python3 scripts/prepare_corpus.py \
  --corpus-id digital-umuganda-tts-rw \
  --out-dir data/processed/digital_umuganda_tts_1k \
  --val-fraction 0.1 \
  --min-line-chars 5

python3 scripts/evaluate_tokenizer_examples.py \
  --manifest data/processed/digital_umuganda_tts_1k/corpora.json \
  --corpus-id digital-umuganda-tts-rw-full \
  --bpe-vocab-size 512 \
  --out-dir experiments/analysis/du_tts_1k_morphology

python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/digital_umuganda_tts_1k/corpora.json \
  --corpus-id digital-umuganda-tts-rw-train \
  --val-corpus-id digital-umuganda-tts-rw-val \
  --tokenizer bpe \
  --tokenizer-fit-scope train-val \
  --bpe-vocab-size 512 \
  --model-config tiny \
  --max-steps 20 \
  --eval-interval 5 \
  --eval-iters 2 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --min-learning-rate 0.0001 \
  --lr-schedule cosine \
  --warmup-steps 2 \
  --grad-clip 1.0 \
  --checkpoint-interval 10 \
  --out-dir experiments/runs/du_tts_1k_tiny_baseline

python3 scripts/create_review_packet.py \
  experiments/runs/du_tts_1k_tiny_baseline
```

Result:

```text
prepared_lines=1000
train_lines=900
val_lines=100
char tokens=83410
BPE tokens=32659
BPE tokens/word=2.8407
initial_val_loss=6.4005
final_val_loss=6.2591
initial_val_perplexity=602.1208
final_val_perplexity=522.7661
interval checkpoints written at steps 10 and 20
```

Interpretation:

This is the first approved-data baseline path. It uses local ignored text
fetched from an approved manifest source, trains on 900 lines, validates on 100
held-out lines, records the learning-rate schedule and gradient clipping, and
writes a sample-review sheet.

The generated sample is still poor, which is expected for a 20-step tiny run.
The useful result is that the pipeline now measures a real held-out
Kinyarwanda validation loss and has the artifacts needed to scale the run.

## 2026-07-05 Resume Smoke Run

Commands:

```bash
python3 scripts/run_track_a_sandbox.py \
  --tokenizer bpe \
  --bpe-vocab-size 64 \
  --max-steps 2 \
  --eval-interval 1 \
  --out-dir experiments/runs/resume_base

python3 scripts/run_track_a_sandbox.py \
  --resume-checkpoint experiments/runs/resume_base/checkpoint.pt \
  --max-steps 2 \
  --eval-interval 1 \
  --out-dir experiments/runs/resume_next
```

Result:

```text
base final_val_loss=4.0818
resume initial_val_loss=4.0783
resume final_val_loss=3.8476
resume checkpoint saved and sampleable
```

Interpretation:

Checkpoint resume now works. The resumed run reuses the saved tokenizer and
model config, loads model weights and optimizer state, then writes a new
checkpoint and run report.

## 2026-07-05 Prepared Split Smoke Run

Commands:

```bash
python3 scripts/prepare_corpus.py \
  --corpus-id toy \
  --out-dir data/processed/toy_smoke

python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/toy_smoke/corpora.json \
  --corpus-id toy-train \
  --val-corpus-id toy-val \
  --tokenizer bpe \
  --tokenizer-fit-scope train-val \
  --bpe-vocab-size 48 \
  --block-size 8 \
  --max-steps 40 \
  --out-dir experiments/runs/prepared_bpe_smoke
```

Result:

```text
prepared_lines=17
train_lines=15
val_lines=2
BPE final_val_loss=3.7307
BPE final_val_perplexity=41.7068
train_tokens=346
val_tokens=19
```

Interpretation:

This is a more honest pipeline check than the original single-file split:
corpus preparation writes separate train/validation files, and the trainer
reads them as separate corpus records. The tiny validation set is unstable, but
the wiring now matches the shape needed for approved data.

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
checkpoint, writes a `run_report.md`, and can generate again from that
checkpoint later.

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
