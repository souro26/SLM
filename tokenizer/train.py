"""
tokenizer/train.py

Trains a BPE tokenizer on Python code streamed from HuggingFace.
Reads configuration from configs/tokenizer.yaml.
Saves artifacts to the configured output directory.

Usage:
    python tokenizer/train.py
    python tokenizer/train.py --config configs/tokenizer.yaml
    python tokenizer/train.py --config configs/tokenizer.yaml --max_files 50000
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import tokenizers
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "tokenizer_train.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load and validate tokenizer config from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded config from %s", config_path)

    required = ["vocab_size", "output_dir", "special_tokens", "min_frequency"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'")

    return cfg


def iter_python_files(cfg: dict, max_files: int, stats: dict):
    """Stream Python source files from configured HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("pip install datasets") from err

    dataset = cfg.get("dataset", "nampdn-ai/tiny-codes")
    lang_field = cfg.get("dataset_language_field", "programming_language")
    lang_value = cfg.get("dataset_language_value", "Python")
    content_field = cfg.get("dataset_content_field", "response")

    log.info("Streaming dataset: %s", dataset)
    log.info("Language filter: %s == %s", lang_field, lang_value)
    log.info("Content field: %s", content_field)
    log.info("Max files: %s", f"{max_files:,}")

    ds = load_dataset(dataset, streaming=True, split="train")

    count = 0
    skipped_lang = 0
    skipped_short = 0
    total_chars = 0

    for sample in ds:
        if sample.get(lang_field) != lang_value:
            skipped_lang += 1
            continue
        content = sample.get(content_field, "")
        if not content or len(content) < 50:
            skipped_short += 1
            continue

        total_chars += len(content)
        yield content

        count += 1
        if count % 10_000 == 0:
            log.info(
                "Streamed %s files | avg length: %.0f chars | skipped short: %s",
                f"{count:,}",
                total_chars / count,
                f"{skipped_short:,}",
            )
        if count >= max_files:
            break

    avg_length = total_chars / count if count else 0
    log.info("Streaming complete.")
    log.info("  Files collected              : %s", f"{count:,}")
    log.info("  Files skipped (wrong lang)   : %s", f"{skipped_lang:,}")
    log.info("  Files skipped (too short)    : %s", f"{skipped_short:,}")
    log.info("  Total characters             : %s", f"{total_chars:,}")
    log.info("  Average file length          : %.0f chars", avg_length)

    stats.update(
        {
            "files_collected": count,
            "files_skipped_language": skipped_lang,
            "files_skipped_short": skipped_short,
            "total_chars": total_chars,
            "avg_file_length": round(avg_length, 1),
        }
    )


def batch_iterator(cfg: dict, max_files: int, stats: dict, batch_size: int = 1000):
    """Yields batches of text strings for the tokenizer trainer."""
    batch = []
    for text in iter_python_files(cfg, max_files, stats):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_tokenizer(cfg: dict) -> Tokenizer:
    """Construct a BPE tokenizer with Python-appropriate settings."""
    special_tokens = cfg["special_tokens"]
    unk_token = next((t for t in special_tokens if "unk" in t), "<|unk|>")
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    log.info("Built BPE tokenizer | unk_token: %s | pre_tokenizer: ByteLevel", unk_token)
    return tokenizer


def build_trainer(cfg: dict) -> trainers.BpeTrainer:
    """Configure the BPE trainer from config."""
    return trainers.BpeTrainer(
        vocab_size=cfg["vocab_size"],
        min_frequency=cfg["min_frequency"],
        special_tokens=cfg["special_tokens"],
        show_progress=cfg.get("show_progress", True),
    )


def add_post_processor(tokenizer: Tokenizer, cfg: dict) -> Tokenizer:
    """Enable padding config for eval/inference batching."""
    pad_token = next((t for t in cfg["special_tokens"] if "pad" in t), "<|pad|>")
    pad_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    log.info("Padding enabled | pad_token: %s | pad_id: %d", pad_token, pad_id)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, cfg: dict, output_dir: Path, training_stats: dict):
    """Save tokenizer.json, vocab.txt, tokenizer_meta.json, and training_log.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir / "tokenizer.json"))
    log.info("Saved tokenizer.json")

    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{idx}\t{token}\n")
    log.info("Saved vocab.txt (%d tokens)", len(sorted_vocab))

    special_tokens = cfg["special_tokens"]
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
        "model_type": "bpe",
        "tokenizers_version": tokenizers.__version__,
    }
    with open(output_dir / "tokenizer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved tokenizer_meta.json")

    training_log = {
        "trained_at": datetime.now(UTC).isoformat(),
        "duration_seconds": training_stats["duration_seconds"],
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": cfg.get("dataset", "nampdn-ai/tiny-codes"),
        "max_files": training_stats["max_files"],
        "files_collected": training_stats.get("files_collected", 0),
        "files_skipped_language": training_stats.get("files_skipped_language", 0),
        "files_skipped_short": training_stats.get("files_skipped_short", 0),
        "total_chars": training_stats.get("total_chars", 0),
        "avg_file_length": training_stats.get("avg_file_length", 0.0),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
        "tokenizers_version": tokenizers.__version__,
        "config": cfg,
    }
    with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    log.info("Saved training_log.json")

    log.info("=" * 50)
    log.info("Tokenizer saved to %s", output_dir)
    log.info("  vocab_size        : %s", f"{tokenizer.get_vocab_size():,}")
    log.info("  duration          : %.1fs", training_stats["duration_seconds"])
    log.info("  files processed   : %s", f"{training_stats.get('files_collected', 0):,}")
    log.info("  total chars       : %s", f"{training_stats.get('total_chars', 0):,}")
    for tok in special_tokens:
        log.info("  %-20s → ID %d", tok, tokenizer.token_to_id(tok))
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on Python corpus")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizer.yaml",
        help="Path to tokenizer config YAML",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Override max_files from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    max_files = args.max_files or cfg.get("max_files", 200_000)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])

    log.info("=" * 50)
    log.info("SLM Tokenizer Training")
    log.info("  config      : %s", args.config)
    log.info("  vocab_size  : %s", f"{cfg['vocab_size']:,}")
    log.info("  max_files   : %s", f"{max_files:,}")
    log.info("  output_dir  : %s", output_dir)
    log.info("=" * 50)

    tokenizer = build_tokenizer(cfg)
    trainer = build_trainer(cfg)

    log.info("Training BPE merges...")
    t_start = time.monotonic()

    stream_stats: dict = {}

    tokenizer.train_from_iterator(
        batch_iterator(cfg, max_files, stream_stats),
        trainer=trainer,
        length=max_files,
    )

    duration = time.monotonic() - t_start
    log.info("Training complete in %.1fs", duration)

    tokenizer = add_post_processor(tokenizer, cfg)

    training_stats = {
        "duration_seconds": round(duration, 2),
        "max_files": max_files,
        **stream_stats,
    }

    save_tokenizer(tokenizer, cfg, output_dir, training_stats)
    log.info("Done. Tokenizer is ready.")


if __name__ == "__main__":
    main()
