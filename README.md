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

## Current Stage

The runnable sandbox now supports two tokenizer paths:

- `char`: character-level baseline,
- `bpe`: small character-seeded BPE tokenizer.

Both paths run through the same tiny causal Transformer training loop and write
the same summary artifacts. Runs are tied to an explicit corpus manifest and can
save checkpoints for later sampling. This still uses toy data, so it proves
wiring and debuggability, not model quality.

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

Prepare a corpus into cleaned train/validation files:

```bash
python3 scripts/prepare_corpus.py \
  --corpus-id toy \
  --out-dir data/processed/toy
```

Train with an explicit prepared validation split:

```bash
python3 scripts/run_track_a_sandbox.py \
  --manifest data/processed/toy/corpora.json \
  --corpus-id toy-train \
  --val-corpus-id toy-val \
  --tokenizer bpe \
  --tokenizer-fit-scope train-val \
  --bpe-vocab-size 48 \
  --block-size 8 \
  --max-steps 40 \
  --out-dir experiments/runs/prepared_bpe_smoke
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
- `checkpoint.pt` unless `--no-save-checkpoint` is passed.

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
- the tokenizer round-trips text with `decode(encode(text))`,
- training loss moves downward on the toy corpus,
- sample generation produces non-empty text,
- the summary file records config, losses, perplexity, tokenizer metadata, and
  sample output.

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

1. Add an approved tiny corpus.
2. Compare char tokenizer vs BPE tokenizer on approved text.
3. Add a short model-card style interpretation.
4. Add a less tiny model config once the data gate passes.
