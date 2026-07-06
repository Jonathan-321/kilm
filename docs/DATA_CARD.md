# KILM Kinyarwanda Corpus Data Card

## Dataset Summary

This corpus is the first large KILM training corpus for a from-scratch
Kinyarwanda causal language model. It contains `22519811` cleaned words and
`34697216` SentencePiece tokens after tokenization.

The tokenized dataset is split into `33985536` train tokens and `711680`
held-out validation tokens using a 98/2 split.

## Sources

| Source | Status | Cleaned words | License / note |
| --- | --- | ---: | --- |
| Digital Umuganda TTS Kinyarwanda text | included | 42848 | local approved import, CC0-1.0 |
| Digital Umuganda MT Kinyarwanda side | included | 121500 | local approved import, CC BY 4.0 |
| Mbaza Kinyarwanda monolingual v01.0 | included | 21715834 | CC BY 4.0 |
| RogerB Kinyarwanda Wikipedia 2023-09-20 | included | 639629 | Wikipedia-derived; attribution/share-alike review required before redistribution |
| Mbaza Kinyarwanda monolingual v01.1 | attempted, not included | 0 | gated on Hugging Face |
| AfriBERTa Gahuza | attempted, not included | 0 | `datasets>=5` rejects the legacy script |
| Masakhane MAFAND en-kin | attempted, not included | 0 | `datasets>=5` rejects the legacy script |

Reference pages:

- https://huggingface.co/datasets/mbazaNLP/kinyarwanda_monolingual_v01.0
- https://huggingface.co/datasets/mbazaNLP/kinyarwanda_monolingual_v01.1
- https://huggingface.co/datasets/RogerB/Kinyarwanda_wikipedia20230920
- https://huggingface.co/datasets/castorini/afriberta-corpus
- https://huggingface.co/datasets/masakhane/mafand

## Processing

The aggregation script normalizes Unicode, removes HTML tags and URLs, strips
replacement characters, rejects lines with excessive symbols or non-Latin text,
requires Kinyarwanda marker words on normal-length lines, removes exact duplicate
lines, and splits very long paragraphs into bounded chunks.

The final corpus is intentionally ignored by git because it is large and may
carry redistribution obligations from upstream sources.

## Known Limitations

The corpus is not fully human-audited. It still contains some mixed-language
proper-name and news metadata patterns. The language filter removes obvious
English spam and standalone footer lines, but it is not a substitute for fluent
speaker review.

Wikipedia-derived data needs license/attribution handling before redistribution.
The model and reports can be used for internal experimentation, but publishing
the dataset or trained artifacts requires a separate license review.

## Intended Use

Use this corpus for internal KILM tokenizer, training-pipeline, and baseline LM
experiments. Do not use it as a learner-facing or production dataset without
additional cleaning, source review, and fluent-speaker evaluation.
