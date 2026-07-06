"""Train and export a SentencePiece tokenizer for KILM."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "kinyarwanda_full_corpus.txt"
DEFAULT_TOKENIZER_DIR = ROOT / "tokenizer"
DEFAULT_PREFIX = DEFAULT_TOKENIZER_DIR / "kinyarwanda_spm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--input-sentence-size", type=int, default=10_000_000)
    parser.add_argument("--shuffle-input-sentence", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(args.corpus)

    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(args.corpus),
        model_prefix=str(args.model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=True,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        user_defined_symbols=[],
    )

    model_path = args.model_prefix.with_suffix(".model")
    vocab_path = args.model_prefix.with_suffix(".vocab")
    export_hf_tokenizer(model_path, args.model_prefix.parent)
    report = {
        "corpus": str(args.corpus),
        "model": str(model_path),
        "vocab": str(vocab_path),
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "model_type": "bpe",
        "byte_fallback": True,
    }
    (args.model_prefix.parent / "tokenizer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"model={model_path}")
    print(f"vocab={vocab_path}")
    print(f"hf_tokenizer={args.model_prefix.parent}")
    return 0


def export_hf_tokenizer(model_path: Path, out_dir: Path) -> None:
    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import SentencePieceExtractor
    from transformers.tokenization_utils_base import generate_merges

    extractor = SentencePieceExtractor(str(model_path))
    proto = extractor.proto
    vocab_scores = [(piece.piece, piece.score) for piece in proto.pieces]
    vocab = {piece: idx for idx, (piece, _score) in enumerate(vocab_scores)}
    merges = generate_merges(vocab, vocab_scores)

    tokenizer_object = Tokenizer(
        BPE(
            vocab=vocab,
            merges=merges,
            unk_token=proto.trainer_spec.unk_piece,
            fuse_unk=True,
            byte_fallback=proto.trainer_spec.byte_fallback,
            dropout=None,
        )
    )

    tokenizer_normalizers = [
        normalizers.Strip(left=False, right=True),
        normalizers.Replace(Regex(" {2,}"), "▁"),
    ]
    if proto.normalizer_spec.precompiled_charsmap:
        tokenizer_normalizers.insert(
            0,
            normalizers.Precompiled(proto.normalizer_spec.precompiled_charsmap),
        )
    tokenizer_object.normalizer = normalizers.Sequence(tokenizer_normalizers)
    tokenizer_object.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer_object.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1),
        ]
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=1024,
    )
    if len(tokenizer) != len(vocab):
        raise RuntimeError(f"bad tokenizer export: {len(tokenizer)} != {len(vocab)}")
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    sys.exit(main())
