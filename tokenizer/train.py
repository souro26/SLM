"""
tokenizer/train.py

Trains a BPE tokenizer on Python code streamed from The Stack v2 (HuggingFace).
No full dataset download — streams and samples what we need, then trains.

Usage:
    python tokenizer/train.py
    python tokenizer/train.py --max_files 100000 --output_dir tokenizer/trained
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2
MAX_FILES = 200000
OUTPUT_DIR = Path("tokenizer/trained")

SPECIAL_TOKENS = [
    "<|endoffile|>",
    "<|pad|>",
    "<|unk|>",
]


def iter_python_files(max_files: int):
    """Stream Python source files from The Stack v2."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    print(f"Streaming The Stack v2 (Python), sampling up to {max_files:,} files...")

    ds = load_dataset(
        "bigcode/the-stack-v2",
        "Python",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0

    for sample in ds:
        content = sample.get("content", "") or sample.get("text", "")
        if not content or len(content) < 50:
            continue
        yield content
        count += 1
        if count % 10000 == 0:
            print(f"  streamed {count:,} files...")
        if count >= max_files:
            break

    print(f"Done streaming. Total files collected: {count:,}")


def batch_iterator(max_files: int, batch_size: int = 1000):
    """Yields batches of text strings for the HuggingFace tokenizer trainer."""
    batch = []
    for text in iter_python_files(max_files):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def build_tokenizer() -> Tokenizer:
    """Instantiate and train a BPE tokenizer for Python."""

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def build_trainer() -> trainers.BpeTrainer:
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    return trainer


def add_post_processor(tokenizer: Tokenizer) -> Tokenizer:
    pad_id = tokenizer.token_to_id("<|pad|>")

    tokenizer.enable_padding(
        pad_id=pad_id,
        pad_token="<|pad|>",
    )

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{idx}\t{token}\n")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "model_type": "bpe",
    }
    with open(output_dir / "tokenizer_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTokenizer saved to {output_dir}/")
    print("  tokenizer.json      — full tokenizer")
    print("  vocab.txt           — human-readable vocab")
    print("  tokenizer_meta.json — special token IDs and metadata")
    print(f"\nVocab size: {tokenizer.get_vocab_size():,}")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok} → ID {tokenizer.token_to_id(tok)}")


def smoke_test(tokenizer: Tokenizer):
    """Quick sanity check after training."""
    samples = [
        "def __init__(self, x: int) -> None:",
        "import numpy as np\nimport torch",
        "    for i in range(len(self.layers)):",
        "self.attention = MultiHeadAttention(d_model, num_heads)",
    ]

    print("\n--- Smoke test ---")
    for s in samples:
        enc = tokenizer.encode(s)
        decoded = tokenizer.decode(enc.ids)
        print(f"\nInput:   {repr(s)}")
        print(f"Tokens:  {enc.tokens}")
        print(f"IDs:     {enc.ids}")
        print(f"Decoded: {repr(decoded)}")
        assert decoded == s, f"Round-trip failed!\nGot: {repr(decoded)}"

    print("\nAll round-trip checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Python corpus")
    parser.add_argument(
        "--max_files",
        type=int,
        default=MAX_FILES,
        help=f"Number of Python files to stream (default: {MAX_FILES})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Where to save the trained tokenizer (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("SLM Tokenizer Training")
    print(f"  vocab_size  : {VOCAB_SIZE:,}")
    print(f"  max_files   : {args.max_files:,}")
    print(f"  output_dir  : {output_dir}")
    print("=" * 60)

    tokenizer = build_tokenizer()
    trainer = build_trainer()

    print("\nTraining BPE merges...")
    tokenizer.train_from_iterator(
        batch_iterator(args.max_files),
        trainer=trainer,
        length=args.max_files,
    )

    tokenizer = add_post_processor(tokenizer)
    save_tokenizer(tokenizer, output_dir)
    smoke_test(tokenizer)

    print("\nDone. Tokenizer is ready.")


if __name__ == "__main__":
    main()
