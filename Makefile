.PHONY: test analyze aggregate train-tokenizer tokenize-full train-full final-report fetch-approved morphology review-packet smoke prepare prepared-smoke compare clean

test:
	PYTHONPATH=src python3 -m pytest -q

analyze:
	python3 scripts/analyze_tokenizers.py \
		--bpe-vocab-size 64 \
		--out-dir experiments/analysis/tokenizers_smoke

aggregate:
	.venv/bin/python scripts/aggregate_corpus.py

train-tokenizer:
	.venv/bin/python scripts/train_sentencepiece.py

tokenize-full:
	.venv/bin/python scripts/tokenize_corpus.py

train-full:
	.venv/bin/python scripts/train_kilm.py

final-report:
	.venv/bin/python scripts/write_final_report.py

fetch-approved:
	python3 scripts/fetch_approved_corpus.py \
		--source digital-umuganda-tts-rw \
		--limit 1000

morphology:
	python3 scripts/evaluate_tokenizer_examples.py \
		--manifest data/processed/digital_umuganda_tts_1k/corpora.json \
		--corpus-id digital-umuganda-tts-rw-full \
		--bpe-vocab-size 512 \
		--out-dir experiments/analysis/du_tts_1k_morphology

review-packet:
	python3 scripts/create_review_packet.py \
		experiments/runs/du_tts_1k_tiny_baseline

smoke:
	python3 scripts/run_track_a_sandbox.py \
		--tokenizer char \
		--max-steps 40 \
		--out-dir experiments/runs/char_smoke
	python3 scripts/run_track_a_sandbox.py \
		--tokenizer bpe \
		--bpe-vocab-size 64 \
		--max-steps 40 \
		--out-dir experiments/runs/bpe_smoke

prepare:
	python3 scripts/prepare_corpus.py \
		--corpus-id toy \
		--out-dir data/processed/toy_smoke

prepared-smoke: prepare
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

compare:
	python3 scripts/compare_runs.py \
		experiments/runs/char_smoke \
		experiments/runs/bpe_smoke \
		--out experiments/analysis/run_comparison.md

clean:
	rm -rf .pytest_cache
	rm -rf data/processed
	rm -rf experiments/analysis experiments/runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
