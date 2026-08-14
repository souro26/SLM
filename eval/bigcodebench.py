"""
eval/bigcodebench.py

BigCodeBench evaluation for the SLM.

BigCodeBench contains ~1,100 complex Python programming problems involving real-world
libraries (e.g. numpy, pandas, requests, matplotlib, scipy).

Requirements:
    pip install bigcodebench datasets

Usage:
    # Generate 1 sample per problem (greedy or temp=0.2)
    python -m eval.bigcodebench --n-samples 1 --temperature 0.0

    # Generate 20 samples for pass@10
    python -m eval.bigcodebench --n-samples 20

    # Skip generation, evaluate existing samples
    python -m eval.bigcodebench --eval-only

Output:
    results/bigcodebench_samples.jsonl — raw completions
    results/bigcodebench_results.json  — pass@k scores
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

LOGS_DIR = Path("logs")
RESULTS_DIR = Path("logs/eval_results")
CHECKPOINT_DIR = Path("checkpoints/pilot-001/step_061000")
MODEL_CONFIG = Path("configs/model.yaml")
TOKENIZER_DIR = Path("tokenizer/trained")
SAMPLES_FILE = RESULTS_DIR / "bigcodebench_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / "bigcodebench_results.json"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "eval_bigcodebench.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


logger = logging.getLogger(__name__)

STOP_STRINGS = ["\ndef ", "\nclass ", "\nif __name__", "\n# ---"]
TEMPERATURE = 0.2
TOP_P = 0.95


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_bigcodebench_problems() -> dict[str, dict]:
    """Load BigCodeBench problems from HuggingFace datasets or bigcodebench package."""
    try:
        from bigcodebench.data import get_bigcodebench

        problems = get_bigcodebench()
        logger.info("Loaded %d BigCodeBench problems via bigcodebench package", len(problems))
        return problems
    except ImportError:
        pass

    try:
        from datasets import load_dataset

        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=True)
        problems = {
            row["task_id"]: {
                "task_id": row["task_id"],
                "prompt": row["complete_prompt"] if "complete_prompt" in row else row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in ds
        }
        logger.info("Loaded %d BigCodeBench problems via HuggingFace datasets", len(problems))
        return problems
    except Exception as e:
        logger.error("Could not load BigCodeBench dataset: %s", e)
        logger.error("Install with: pip install bigcodebench datasets")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_samples(
    problems: dict[str, dict],
    n_samples: int,
    checkpoint_dir: Path,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[dict]:
    from eval.generator import CodeGenerator

    logger.info("Loading model from %s...", checkpoint_dir)
    gen = CodeGenerator(
        checkpoint_dir=checkpoint_dir,
        model_config_path=MODEL_CONFIG,
        tokenizer_dir=TOKENIZER_DIR,
    )

    samples: list[dict] = []
    total = len(problems)

    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem.get("complete_prompt") or problem["prompt"]
        logger.info("[%d/%d] %s — generating %d sample(s)...", i + 1, total, task_id, n_samples)
        t0 = time.time()

        completions = gen.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=STOP_STRINGS,
            num_samples=n_samples,
        )
        logger.info("  done in %.1fs", time.time() - t0)

        for completion in completions:
            samples.append({"task_id": task_id, "solution": completion})

    return samples


def save_samples(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Evaluation via BigCodeBench CLI
# ---------------------------------------------------------------------------


def run_bigcodebench_eval(samples_path: Path) -> None:
    """Run official bigcodebench evaluation CLI."""
    cmd = [
        sys.executable,
        "-m",
        "bigcodebench.evaluate",
        "--samples",
        str(samples_path),
    ]
    logger.info("Running BigCodeBench evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning("BigCodeBench evaluation command exited with code %d", result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="BigCodeBench benchmark for SLM")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--samples-file", type=str, default=str(SAMPLES_FILE))
    args = parser.parse_args()

    samples_path = Path(args.samples_file)

    if not args.eval_only:
        logger.info("=== BigCodeBench Generation ===")
        logger.info("n_samples   : %d", args.n_samples)
        logger.info("temperature : %.2f", args.temperature)
        logger.info("checkpoint  : %s", args.checkpoint)

        problems = load_bigcodebench_problems()
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
        logger.info("Skipping generation, using %s", samples_path)

    logger.info("")
    logger.info("=== BigCodeBench Evaluation ===")
    run_bigcodebench_eval(samples_path)


if __name__ == "__main__":
    main()
