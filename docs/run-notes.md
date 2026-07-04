# Track A Sandbox Run Notes

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

Replace the character tokenizer with the team BPE tokenizer and rerun this same
kind of sandbox on a small approved corpus.
