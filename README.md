# KILM

KILM is a standalone learning and feasibility sandbox for the from-scratch
Kinyarwanda language-model track. It is separate from the main `kinyalm`
planning repo, separate from the CS336 assignment repos, and separate from the
final training path.

The question this sandbox answers is:

```text
Can we run the whole Track A loop end to end before real data and larger models?
```

The loop is:

```text
text corpus
→ tokenizer
→ token IDs
→ training batches
→ tiny causal language model
→ loss curve
→ sample generation
→ written interpretation
```

## Production-Scale Baseline Path

The repo now includes a larger from-scratch Kinyarwanda LM path in addition to
the original tiny sandbox loop:

```bash
make aggregate
make train-tokenizer
make tokenize-full
make train-full
make final-report
```

That path aggregates the large Kinyarwanda corpus, trains a 32k SentencePiece
BPE tokenizer with byte fallback, tokenizes into 1024-token Arrow blocks, and
trains a 109M-parameter LLaMA-style causal LM from scratch.

The training script defaults to a 50000-step target with AdamW, cosine LR,
2000 warmup steps, gradient clipping, checkpointing, validation, and sample
generation every 2000 steps. On local Apple MPS, the recorded baseline completed
2000 steps as a compute-limited run; see `docs/FINAL_RUN_REPORT.md`,
`docs/DATA_CARD.md`, and `docs/MODEL_CARD.md`.

Generated text is still not fluent enough for product use. Treat the current
model as proof that the pipeline works, not proof that the model is ready.

## Conversation SFT Pivot

For a usable Kinyarwanda assistant, the next path is supervised fine-tuning
on real user/assistant conversations, then human-rated evaluation. Prepare
conversation data:

```bash
python scripts/prepare_sft_conversations.py \
  --input data/sft/raw_conversations.jsonl \
  --out-dir data/sft/processed \
  --validation-fraction 0.1
```

Bootstrap translation-style SFT pairs from open Hugging Face datasets:

```bash
python scripts/generate_sft_bootstrap.py
```

Run QLoRA SFT against a stronger open instruct base model:

```bash
python scripts/train_sft_qlora.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --train-file data/sft/processed/train.jsonl \
  --validation-file data/sft/processed/validation.jsonl \
  --output-dir checkpoints/sft/qwen2.5-7b-kinyarwanda-qlora
```

Evaluate any base or SFT model with a held-out benchmark:

```bash
python scripts/run_conversation_benchmark.py \
  --model checkpoints/kilm-llama-100m/checkpoint-50000 \
  --benchmark data/eval/kinyarwanda_conversation_benchmark.jsonl
```

## Current Stage

The runnable sandbox now supports approved-corpus baseline runs, not only toy
smoke tests. It can fetch approved source text into local ignored files,
prepare train/validation splits, compare tokenizers, evaluate morphology-focused
examples, train a tiny baseline with a learning-rate schedule, write interval
checkpoints, and generate data/model cards plus a sample-review sheet.

Two tokenizer paths are available:

- `char`: character-level baseline,
- `bpe`: small character-seeded BPE tokenizer.

Both paths run through the same tiny causal Transformer training loop and write
the same summary artifacts. Runs are tied to an explicit corpus manifest and
can save checkpoints for later sampling. The approved-data path now gives a real
baseline signal, but it is still not a useful model-quality claim until longer
training and fluent-speaker review happen.

## Why Keep Character-Level

Character-level tokenization is not the goal, but it is still useful because:

- it gives BPE a baseline,
- it makes encode/decode behavior easy to inspect,
- it lets us separate model-loop bugs from tokenizer bugs.

## Run

From the repo root:

```bash
python3 scripts/run_track_a_sandbox.py --max-steps 40
```

Run the test suite:

```bash
make test
```

Run the BPE path:

```bash
python3 scripts/run_track_a_sandbox.py \
  --tokenizer bpe \
  --bpe-vocab-size 64 \
  --max-steps 40 \
  --out-dir experiments/runs/bpe_smoke
```

Compare tokenizers without training:

```bash
python3 scripts/analyze_tokenizers.py \
  --bpe-vocab-size 64 \
  --out-dir experiments/analysis/tokenizers_smoke
```

Fetch the approved Digital Umuganda TTS sentence subset into local ignored data:

```bash
python3 scripts/fetch_approved_corpus.py \
  --source digital-umuganda-tts-rw \
  --limit 1000
```

Prepare a corpus into cleaned train/validation files:

```bash
python3 scripts/prepare_corpus.py \
  --corpus-id digital-umuganda-tts-rw \
  --out-dir data/processed/digital_umuganda_tts_1k \
  --val-fraction 0.1 \
  --min-line-chars 5
```

Train with an explicit prepared validation split:

```bash
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
  --device auto \
  --out-dir experiments/runs/du_tts_1k_tiny_baseline
```

For a stronger local Apple GPU baseline, fetch the full TTS source without
`--limit`, prepare it under `data/processed/digital_umuganda_tts_full`, then
run:

```bash
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
```

Use `--device auto` by default. It prefers CUDA, then Apple MPS, then CPU.

Run morphology-focused tokenizer evaluation:

```bash
python3 scripts/evaluate_tokenizer_examples.py \
  --manifest data/processed/digital_umuganda_tts_1k/corpora.json \
  --corpus-id digital-umuganda-tts-rw-full \
  --bpe-vocab-size 512 \
  --out-dir experiments/analysis/du_tts_1k_morphology
```

Generate the data card, model card, and sample review sheet for a run:

```bash
python3 scripts/create_review_packet.py \
  experiments/runs/du_tts_1k_tiny_baseline
```

Sample from a saved checkpoint:

```bash
python3 scripts/sample_checkpoint.py \
  experiments/runs/bpe_smoke/checkpoint.pt \
  --prompt Muraho \
  --sample-tokens 80
```

Resume training from a checkpoint:

```bash
python3 scripts/run_track_a_sandbox.py \
  --tokenizer bpe \
  --bpe-vocab-size 64 \
  --resume-checkpoint experiments/runs/bpe_smoke/checkpoint.pt \
  --max-steps 20 \
  --out-dir experiments/runs/bpe_resume_smoke
```

Compare completed runs:

```bash
python3 scripts/compare_runs.py \
  experiments/runs/char_smoke \
  experiments/runs/bpe_smoke \
  --out experiments/analysis/run_comparison.md
```

The common local workflow is also available as:

```bash
make fetch-approved
make morphology
make review-packet
make analyze
make smoke
make prepared-smoke
make compare
```

Outputs are written to:

```text
experiments/runs/track_a_sandbox/
```

Each run writes:

- `summary.json`,
- `run_report.md`,
- `sample.txt`,
- `vocab.json`,
- `tokenizer.json`.
- `checkpoint.pt` unless `--no-save-checkpoint` is passed,
- `checkpoint_step_*.pt` when `--checkpoint-interval` is set.

Experiment output folders are local artifacts and should not be committed by
default.

## Corpus Manifest

Known corpora are declared in:

```text
data/corpora.json
```

The default `toy` corpus is allowed for smoke tests. Any direct `--corpus` path
or manifest entry with a non-`approved`/non-`toy` status is blocked unless
`--allow-unapproved-corpus` is passed. That escape hatch is for local debugging,
not for claiming training data is approved.

Approved source text under `data/approved/` and raw downloads under `data/raw/`
are local artifacts and are ignored by Git. Re-fetch them from manifest/source
definitions when needed.

Prepared corpus outputs under `data/processed/` are local artifacts by default.
Each prepared folder contains `full.txt`, `train.txt`, `val.txt`, `stats.json`,
`corpora.json`, and `corpus_card.md`.

For explicit train/validation runs, `--tokenizer-fit-scope train-val` fits the
tokenizer on both splits so tiny smoke tests do not fail on unseen validation
characters. Use `--tokenizer-fit-scope train` when you want a stricter check.

## Repository Boundary

This repo is intentionally small and disposable. Keep toy data, learning notes,
and runnable feasibility checks here. Move only validated decisions and short
status summaries back into the main `kinyalm` planning repo.

## Success Criteria For This Sandbox

The sandbox is successful if:

- the script runs from a fresh checkout,
- approved source text can be fetched into an ignored local corpus,
- the tokenizer round-trips text with `decode(encode(text))`,
- BPE compression and morphology splits are inspectable on approved text,
- validation loss/perplexity are measured on held-out approved Kinyarwanda text,
- learning-rate schedule, gradient clipping, and checkpoint intervals are
  recorded in run metadata,
- sample generation produces non-empty text,
- the summary file records config, losses, perplexity, tokenizer metadata, and
  sample output,
- the run can produce a data card, model card, and speaker review sheet.

This does not mean the final Kinyarwanda LM is successful. It only means the
learning pipeline is wired together.

## Track A Gates

Track A can replace or reduce the need for Track B only if later stages pass
real gates:

1. Approved corpus exists.
2. BPE tokenizer is implemented and evaluated.
3. Tiny model trains reproducibly.
4. Validation loss improves on held-out Kinyarwanda text.
5. Samples are reviewed by fluent speakers.
6. The model can support learning tasks better than retrieval/prompting alone.

Until those gates pass, Track B remains a fallback for usefulness.

## Next Sandbox Stages

1. Run the same path on the full approved TTS text and the approved MT
   Kinyarwanda side.
2. Move from `tiny` to `small` or `baseline_gpu` once GPU memory is confirmed.
3. Increase steps and evaluation intervals enough for a meaningful loss curve.
4. Send `sample_review.tsv` to fluent speakers and fold the feedback into the
   model card.
5. Decide whether the next training run needs more corpus cleanup before scale.
