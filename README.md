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

The current runnable sandbox uses a character-level tokenizer. That is not the
final tokenizer. It is a safe first step because it lets the team understand the
full LM workflow before implementing and validating BPE.

## Why Character-Level First

Character-level tokenization is not the goal, but it is useful because:

- it avoids unapproved external data,
- it avoids pretending the BPE tokenizer is done,
- it makes encode/decode behavior easy to inspect,
- it lets us test the training loop with a tiny corpus,
- it gives the team a baseline before replacing it with BPE.

## Run

From the repo root:

```bash
python3 scripts/run_track_a_sandbox.py --max-steps 40
```

Outputs are written to:

```text
experiments/runs/track_a_sandbox/
```

That folder is local experiment output and should not be committed by default.

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
- the summary file records config, losses, and sample output.

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

1. Add BPE tokenizer path.
2. Add an approved tiny corpus.
3. Compare char tokenizer vs BPE tokenizer.
4. Add validation split and perplexity report.
5. Add a short model-card style interpretation.
