"""
eval/humaneval.py

HumanEval and HumanEval+ benchmark evaluation for the SLM.

Generates N completions for each of the 164 HumanEval problems, saves them
to results/humaneval_samples.jsonl, then runs the official pass@k evaluation.

Requirements (run once before using this script):
    pip install human-eval datasets

Usage:
    # Quick run — 20 samples/problem, reports pass@1 and pass@10
    python -m eval.humaneval

    # More samples for better pass@100 estimate (takes longer)
    python -m eval.humaneval --n-samples 100

    # Use a specific checkpoint
    python -m eval.humaneval --checkpoint checkpoints/pilot-001/step_061000

    # Skip generation if you already have samples, just re-evaluate
    python -m eval.humaneval --eval-only

Output:
    results/humaneval_samples.jsonl  — raw completions
    results/humaneval_results.json   — pass@k scores
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/pilot-001/step_061000")
MODEL_CONFIG = Path("configs/model.yaml")
TOKENIZER_DIR = Path("tokenizer/trained")
RESULTS_DIR = Path("results")
SAMPLES_FILE = RESULTS_DIR / "humaneval_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / "humaneval_results.json"

STOP_STRINGS = ["\ndef ", "\nclass ", "\nif __name__", "\n# ---"]
TEMPERATURE = 0.8
TOP_P = 0.95


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_humaneval_problems() -> list[dict]:
    """Load the 164 HumanEval problems from the official package or HuggingFace."""
    try:
        from human_eval.data import read_problems

        problems = list(read_problems().values())
        logger.info("Loaded %d problems from human-eval package", len(problems))
        return problems
    except ImportError:
        pass

    try:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
        problems = [
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in ds
        ]
        logger.info("Loaded %d problems from HuggingFace datasets", len(problems))
        return problems
    except ImportError:
        pass

    logger.error(
        "Could not load HumanEval. Install with:\n"
        "  pip install human-eval datasets\n"
        "or:\n"
        "  pip install git+https://github.com/openai/human-eval.git"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_samples(
    problems: list[dict],
    n_samples: int,
    checkpoint_dir: Path,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict]:
    """Generate n_samples completions for each problem."""
    from eval.generator import CodeGenerator

    logger.info("Loading model from %s...", checkpoint_dir)
    gen = CodeGenerator(
        checkpoint_dir=checkpoint_dir,
        model_config_path=MODEL_CONFIG,
        tokenizer_dir=TOKENIZER_DIR,
    )

    samples: list[dict] = []
    total = len(problems)

    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        logger.info(
            "[%d/%d] %s — generating %d sample(s)...",
            i + 1,
            total,
            task_id,
            n_samples,
        )
        t0 = time.time()

        completions = gen.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=STOP_STRINGS,
            num_samples=n_samples,
        )

        elapsed = time.time() - t0
        logger.info("  done in %.1fs", elapsed)

        for completion in completions:
            samples.append({"task_id": task_id, "completion": completion})

    return samples


def save_samples(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(samples_path: Path, k_values: list[int]) -> dict:
    """Run the official HumanEval functional correctness evaluation."""
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        logger.error(
            "human-eval package not found. Install with:\n"
            "  pip install git+https://github.com/openai/human-eval.git"
        )
        sys.exit(1)

    logger.info("Running functional correctness evaluation...")
    logger.info("(This executes the generated code — may take a few minutes)")

    results = evaluate_functional_correctness(str(samples_path), k=k_values)

    logger.info("")
    logger.info("=" * 50)
    for key, value in results.items():
        logger.info("  %-20s %.4f  (%.1f%%)", key, value, value * 100)
    logger.info("=" * 50)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="HumanEval benchmark for SLM")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Completions per problem. 20=pass@10, 200=pass@100 (default: 20)",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip generation; evaluate existing samples file",
    )
    parser.add_argument("--samples-file", type=str, default=str(SAMPLES_FILE))
    args = parser.parse_args()

    samples_path = Path(args.samples_file)
    k_values = [k for k in [1, 10, 100] if k <= args.n_samples] or [1]

    if not args.eval_only:
        logger.info("=== HumanEval Generation ===")
        logger.info("n_samples    : %d", args.n_samples)
        logger.info("temperature  : %.2f", args.temperature)
        logger.info("top_p        : %.2f", args.top_p)
        logger.info("checkpoint   : %s", args.checkpoint)
        logger.info("")

        problems = load_humaneval_problems()
        samples = generate_samples(
            problems=problems,
            n_samples=args.n_samples,
            checkpoint_dir=Path(args.checkpoint),
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        save_samples(samples, samples_path)
    else:
        if not samples_path.exists():
            logger.error("Samples file not found: %s", samples_path)
            sys.exit(1)
        logger.info("Skipping generation, loading from %s", samples_path)

    logger.info("")
    logger.info("=== HumanEval Evaluation (pass@%s) ===", k_values)
    results = run_evaluation(samples_path, k_values)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "benchmark": "HumanEval",
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pass_at_k": dict(results),
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
