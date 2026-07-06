# KILM LLaMA 100M Baseline Model Card

## Model Summary

This is a from-scratch Kinyarwanda causal language model baseline trained with
the rebuilt KILM pipeline. It is a LLaMA-style decoder-only Transformer with
`109529856` parameters, a `32000`-token SentencePiece BPE tokenizer, and a
`1024` token context length.

This model is not production-ready. The 2,000-step local baseline shows that the
pipeline works and that validation perplexity improves, but generated samples
remain repetitive and semantically unreliable.

## Architecture

- Model family: LLaMA-style causal LM
- Layers: 12
- Attention heads: 12
- Hidden size: 768
- Intermediate size: 2048
- Context length: 1024
- Dropout: 0.1 attention dropout
- Vocabulary: 32000
- Tokenizer: SentencePiece BPE with byte fallback
- Attention implementation: PyTorch scaled dot-product attention through
  Transformers

## Training Run

- Dataset: `data/tokenized/kinyarwanda_spm_1024`
- Train tokens: `33985536`
- Validation tokens: `711680`
- Optimizer: AdamW
- LR schedule: cosine with 2000 warmup steps
- Gradient clipping: max norm 1.0
- Completed local run: 2000 steps
- Default script target: 50000 steps
- Device used: Apple MPS, full precision

## Metrics

- Final train loss: `6.818963836669922`
- Final train perplexity: `915.036391549543`
- Final validation loss: `5.864002704620361`
- Final validation perplexity: `352.13080251279723`

The untrained smoke run had validation perplexity around `37421.9283`, so the
2,000-step baseline demonstrates real learning. It does not demonstrate fluent
generation.

## Sample Quality

Generated text is Kinyarwanda-shaped and uses common Kinyarwanda function words,
but it repeats phrases, invents numbers, and does not reliably maintain meaning.
The current decision is `needs-more-training-and-fluent-review`.

## Intended Use

Use this checkpoint to validate the rebuilt data, tokenizer, model, resume,
checkpoint, validation, and reporting path. It should not be used for
learner-facing text generation, translation, tutoring, grading, or public
deployment.

## Next Steps

Run a longer continuation from `checkpoints/kilm-llama-100m/checkpoint-2000` on
stronger CUDA hardware, keep checkpoint/eval/sample intervals at 2000 steps, and
send samples to fluent Kinyarwanda speakers before treating any model output as
usable.
