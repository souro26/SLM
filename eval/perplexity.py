"""
eval/perplexity.py

Measures perplexity of the trained SLM on arbitrary Python source files.

Perplexity = exp(average cross-entropy loss over all tokens).
Lower is better. Our validation loss at the end of training was ~1.3-1.6,
so perplexity on similar held-out Python code should be around 3-5.

Usage:
    # Evaluate on a single file
    python -m eval.perplexity --files path/to/file.py

    # Evaluate on all .py files in a directory (recursively)
    python -m eval.perplexity --dirs path/to/repo/

    # Mix both, set stride
    python -m eval.perplexity --dirs path/to/repo/ --stride 512

Output:
    Prints per-file perplexity and overall aggregate perplexity.
    Also saves results to results/perplexity.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch

from model.config import ModelConfig
from model.transformer import TransformerModel
from tokenizer.tokenizer import SLMTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/pilot-001/step_061000")
MODEL_CONFIG = Path("configs/model.yaml")
TOKENIZER_DIR = Path("tokenizer/trained")
RESULTS_DIR = Path("results")


def load_model_and_tokenizer(device: torch.device):
    logger.info("Loading model config...")
    cfg = ModelConfig.from_yaml(str(MODEL_CONFIG))

    logger.info("Building model...")
    model = TransformerModel(cfg)

    tensor_path = CHECKPOINT_DIR / "tensors.pt"
    logger.info("Loading weights from %s", tensor_path)
    state = torch.load(tensor_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    logger.info("Loading tokenizer from %s", TOKENIZER_DIR)
    tokenizer = SLMTokenizer(TOKENIZER_DIR)
    return model, tokenizer, cfg


@torch.inference_mode()
def compute_perplexity(
    model: TransformerModel,
    tokenizer: SLMTokenizer,
    text: str,
    context_length: int,
    stride: int,
    device: torch.device,
) -> tuple[float, int]:
    """
    Compute perplexity using a sliding window over the tokenized text.

    Uses stride < context_length so every token (except the very first
    context window's prefix) is predicted with maximum context.

    Returns (perplexity, num_tokens_evaluated).
    """
    token_ids = tokenizer.encode(text, add_eof=True)
    if len(token_ids) < 2:
        return float("nan"), 0

    total_nll = 0.0
    total_tokens = 0

    seq = torch.tensor(token_ids, dtype=torch.long, device=device)

    for begin in range(0, len(token_ids) - 1, stride):
        end = min(begin + context_length, len(token_ids))
        chunk = seq[begin:end]

        if len(chunk) < 2:
            break

        input_ids = chunk[:-1].unsqueeze(0)
        target_ids = chunk[1:].unsqueeze(0)

        logits, _ = model(input_ids)

        # Only count tokens in the non-overlapping ("new") region of this window
        # so each token is evaluated exactly once with maximum context.
        count_from = 0 if begin == 0 else max(0, len(target_ids[0]) - stride)

        loss = torch.nn.functional.cross_entropy(
            logits[:, count_from:, :].reshape(-1, logits.size(-1)),
            target_ids[:, count_from:].reshape(-1),
            reduction="sum",
        )
        n_new = int(target_ids[:, count_from:].numel())

        total_nll += loss.item()
        total_tokens += n_new

        if end == len(token_ids):
            break

    if total_tokens == 0:
        return float("nan"), 0

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll), total_tokens


def collect_files(dirs: list[str], files: list[str]) -> list[Path]:
    paths: list[Path] = []
    for d in dirs:
        paths.extend(sorted(Path(d).rglob("*.py")))
    for f in files:
        p = Path(f)
        if p.exists():
            paths.append(p)
        else:
            logger.warning("File not found: %s", f)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SLM perplexity on Python files")
    parser.add_argument("--files", nargs="*", default=[], help="Individual .py files")
    parser.add_argument("--dirs", nargs="*", default=[], help="Directories to scan recursively")
    parser.add_argument(
        "--stride", type=int, default=512, help="Sliding-window stride (default 512)"
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR), help="Checkpoint directory")
    args = parser.parse_args()

    if not args.files and not args.dirs:
        parser.error("Provide at least one --files or --dirs argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_model_and_tokenizer(device)
    context_length = cfg.context_length

    paths = collect_files(args.dirs, args.files)
    if not paths:
        logger.error("No .py files found.")
        sys.exit(1)

    logger.info("Evaluating perplexity on %d file(s)...", len(paths))

    results: list[dict] = []
    total_nll_sum = 0.0
    total_tok_sum = 0

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            continue

        ppl, n_tok = compute_perplexity(model, tokenizer, text, context_length, args.stride, device)

        if math.isnan(ppl) or n_tok == 0:
            logger.warning("Skipping %s (too short)", path)
            continue

        logger.info("  %-60s  ppl=%6.2f  tokens=%d", str(path)[-60:], ppl, n_tok)
        results.append({"file": str(path), "perplexity": ppl, "tokens": n_tok})

        total_nll_sum += math.log(ppl) * n_tok
        total_tok_sum += n_tok

    if total_tok_sum == 0:
        logger.error("No tokens evaluated.")
        sys.exit(1)

    aggregate_ppl = math.exp(total_nll_sum / total_tok_sum)
    logger.info("")
    logger.info("=" * 70)
    logger.info("Aggregate perplexity : %.4f", aggregate_ppl)
    logger.info("Total tokens         : %d", total_tok_sum)
    logger.info("Files evaluated      : %d", len(results))
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "aggregate_perplexity": aggregate_ppl,
        "total_tokens": total_tok_sum,
        "files_evaluated": len(results),
        "per_file": results,
    }
    out_path = RESULTS_DIR / "perplexity.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
