# Final KILM Run Report

## Dataset

- Corpus path: `/Users/jonathanmuhire/Documents/kilm/data/kinyarwanda_full_corpus.txt`
- Corpus words: `22519811`
- Corpus lines: `192966`
- Tokenized train tokens: `33985536`
- Tokenized validation tokens: `711680`
- Total tokenized tokens: `34697216`
- Split: `98% train / 2% validation`

## Sources

- digital-umuganda-tts-rw: `42848` cleaned words
- digital-umuganda-mt-rw: `121500` cleaned words
- mbaza-monolingual-v01.1: not included (`failed`; Large preferred version; gated on Hugging Face, so this may fail without access.)
- mbaza-monolingual-v01.0: `21715834` cleaned words
- rogerb-wikipedia-20230920: `639629` cleaned words
- castorini-afriberta-gahuza: not included (`failed`; AfriBERTa corpus; may be unavailable in datasets>=5 because it uses a legacy script.)
- masakhane-mafand-en-kin: not included (`failed`; Masakhane MAFAND news translation data; may be unavailable in datasets>=5.)

## Model

- Model directory: `/Users/jonathanmuhire/Documents/kilm/checkpoints/kilm-llama-100m`
- Parameters: `109529856`
- Architecture: LLaMA-style causal LM, 12 layers, 12 attention heads, hidden size 768, context length 1024
- Tokenizer: SentencePiece BPE, 32000 vocabulary size, byte fallback enabled

## Run Scope

- Script default: `50000` max steps
- Completed local baseline: `2000` steps
- Reason: local Apple MPS compute can run the pipeline end to end, but a 50000-step run would take substantially longer than this session.
- Checkpoint/eval/sample interval used: `2000` steps
- Mixed precision: disabled on local MPS; CUDA runs use bf16/fp16 when available.

## Metrics

- Final train loss: `6.818963836669922`
- Final train perplexity: `915.036391549543`
- Final validation loss: `5.864002704620361`
- Final validation perplexity: `352.13080251279723`

## Greedy Samples

### Sample 1

```text
Muraho mu gihe cy’uko hari abantu mu gihe cy’uko mu gihe cy’uko hari n’abandi. Ati “Iyo ndwara, “Iyo ndwara, “Iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu,
```

### Sample 2

```text
Muraho mu gihe cy’uko hari abantu mu gihe cy’uko mu gihe cy’uko hari n’abandi. Ati “Iyo ndwara, “Iyo ndwara, “Iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu,
```

### Sample 3

```text
Muraho mu gihe cy’uko hari abantu mu gihe cy’uko mu gihe cy’uko hari n’abandi. Ati “Iyo ndwara, “Iyo ndwara, “Iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu,
```

### Sample 4

```text
Muraho mu gihe cy’uko hari abantu mu gihe cy’uko mu gihe cy’uko hari n’abandi. Ati “Iyo ndwara, “Iyo ndwara, “Iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu,
```

### Sample 5

```text
Muraho mu gihe cy’uko hari abantu mu gihe cy’uko mu gihe cy’uko hari n’abandi. Ati “Iyo ndwara, “Iyo ndwara, “Iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko iyo ndwara, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu, ariko ngo n’abandi bantu,
```


## Temperature 0.7 / Top-k 40 Samples

### Sample 1

```text
Muraho aho ngo bibaza ko umuntu wese nta babyeyi”. 1-2:22. 5.19:1.276-10-133, 43.4192 69922). 19.1.1. 21:48-19. 67. 22664] Nanone Yehova na nyina wige? 193 (11) 1973.08. Ese Yehova? Ese Yehova? Ese Yehova, Dawidi Yehova yari Yehova afite akamaro
```

### Sample 2

```text
Muraho, ngo abaza iki mu gihe ku muntu wo ku buryo. Ati “Mu myaka by’iki nzego ya Polisi ya mbere y’u Rwanda muri Afurika ku mukino wa kabiri, ya kabiri n’ibigo by’ubushakashatsi bw’abo bantu, ariko ngo iyo ndwara ku isi ku rwego rw’abo mu bibazo, no kubona ko hari hari n’abandi bikorwa by’umutekano, ariko ngo babashe kwigisha, kuko hari n’abantu b’uko zidufasha mu miryango yabo. Ati “Abantu n’abandi barindwi mu Karere ka Rubavu, ariko ngo ntibashe, kuko ngo n’abo baburwe,
```

### Sample 3

```text
Muraho y’imyaka icumi ya Leta ya buri kwezi ku giti cye n’u Rwanda ndetse n’ubwo abahuza ko ari ugukorerwa cyangwa ibyo bice byo mu gukora ko buri gihe abandi mu gihe, uko n’abandi bagwiriye. Yagize ati “Twabona ko abantu ari ukuri, ariko ngo ntangiye kuba bukugisha, kuko abantu bari mu buryo, ariko ibyo ngo mu gihe cy’abana mu buryo, ariko hari n’abo bantu n’aho bitegura ku buryo bw’uko hari icyo kibazo kuko iyo ngingo. Yagize ati “Niba hari hari abantu mu mutwe, ariko iyo n’
```

### Sample 4

```text
Muraho, kandi n’uko u Rwanda n’abandi bahanzi babiri ngo bakaba mu Rwanda. Agira ati “Icyo gihe hari abandi, hari abantu, kandi n’ubwo hari n’abo bari, kandi nta cyo gukora mu bikorwa by’uko hari n’ibindi n’ubwo twari ari ngombwa”. Umuyobozi Mukuru wa Polisi y’imari myiza, Dr, avuga ko mu gihe cyasohotse mu bikorwa bya Perezida Kagame, avuga ko mu gihe cya mbere ko kugeza ubu bakaba ku bantu, ubu bakaba bari mu kigo cy’u Rwanda muri Jenoside mu Karere ka Nyamagabe. Yavuze ko ibi bikorwa by’abantu bari mu nzego
```

### Sample 5

```text
Muraho.
```
