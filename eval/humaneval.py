"""
eval/humaneval.py

HumanEval and HumanEval+ benchmark evaluation for the SLM.

Generates N completions for each of the 164 HumanEval problems, saves them
to logs/eval_results/humaneval_samples.jsonl, then runs functional correctness.

Uses native subprocess execution for 100% Windows compatibility (avoiding
Unix-only signal.setitimer crashes in OpenAI's original runner).

Requirements:
    pip install datasets

Usage:
    # 1 sample per problem (greedy pass@1)
    python -m eval.humaneval --n-samples 1 --temperature 0.0

    # 10 samples per problem for pass@1 and pass@10
    python -m eval.humaneval --n-samples 10 --temperature 0.2

    # Score existing samples file without re-generating
    python -m eval.humaneval --eval-only

Output:
    logs/eval_results/humaneval_samples.jsonl  — raw completions
    logs/eval_results/humaneval_results.json   — pass@k scores
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

LOGS_DIR = Path("logs")
RESULTS_DIR = Path("logs/eval_results")
CHECKPOINT_DIR = Path("checkpoints/pilot-001/step_061000")
MODEL_CONFIG = Path("configs/model.yaml")
TOKENIZER_DIR = Path("tokenizer/trained")
SAMPLES_FILE = RESULTS_DIR / "humaneval_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / "humaneval_results.json"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "eval_humaneval.log"
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
EXEC_TIMEOUT = 5.0


def _ensure_indented_prompt(prompt: str) -> str:
    """Ensure prompt ends with 4-space indentation for base model completion."""
    prompt_stripped = prompt.rstrip()
    if (
        prompt_stripped.endswith(":")
        or prompt_stripped.endswith('"""')
        or prompt_stripped.endswith("'''")
    ):
        return prompt_stripped + "\n    "
    if not prompt.endswith("    "):
        return prompt_stripped + "\n    "
    return prompt


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_humaneval_problems() -> list[dict]:
    """Load the 164 HumanEval problems."""
    try:
        from human_eval.data import read_problems

        probs = read_problems()
        problems = [probs[k] for k in sorted(probs.keys(), key=lambda x: int(x.split("/")[1]))]
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

    logger.error("Could not load HumanEval dataset. Install datasets or human-eval.")
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
        raw_prompt = problem["prompt"]
        prompt = _ensure_indented_prompt(raw_prompt)

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

        logger.info("  done in %.1fs", time.time() - t0)

        for completion in completions:
            samples.append(
                {
                    "task_id": task_id,
                    "raw_prompt": raw_prompt,
                    "prompt": prompt,
                    "completion": completion.lstrip(" "),
                }
            )

    return samples


def save_samples(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Execution & Evaluation (Windows-safe Subprocess Execution)
# ---------------------------------------------------------------------------


def _run_code_in_subprocess(code: str, timeout: float) -> tuple[bool, str]:
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode == 0:
            return True, ""
        return False, (res.stderr or res.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)


def evaluate_sample(sample: dict, problem: dict, timeout: float = EXEC_TIMEOUT) -> bool:
    """Reconstruct full code and execute problem's unit tests."""
    prompt = sample.get("prompt") or _ensure_indented_prompt(problem["prompt"])
    completion = sample["completion"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    passed, _ = _run_code_in_subprocess(full_code, timeout)
    return passed


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_evaluation(
    samples_path: Path, problems: list[dict], k_values: list[int], timeout: float = EXEC_TIMEOUT
) -> dict:
    samples_by_task: dict[str, list[dict]] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples_by_task.setdefault(obj["task_id"], []).append(obj)

    prob_map = {p["task_id"]: p for p in problems}
    pass_at_k_accum: dict[int, list[float]] = {k: [] for k in k_values}

    total_passed_tasks = 0

    for tid, problem in prob_map.items():
        sample_list = samples_by_task.get(tid, [])
        n = len(sample_list)
        if n == 0:
            continue

        c = sum(evaluate_sample(s, problem, timeout) for s in sample_list)
        if c > 0:
            total_passed_tasks += 1
            logger.info("  %-15s PASSED (%d/%d)", tid, c, n)
        else:
            logger.info("  %-15s FAILED (0/%d)", tid, n)

        for k in k_values:
            if k <= n:
                pass_at_k_accum[k].append(compute_pass_at_k(n, c, k))

    results = {}
    logger.info("")
    logger.info("=" * 50)
    for k in k_values:
        vals = pass_at_k_accum[k]
        score = sum(vals) / len(vals) if vals else 0.0
        results[f"pass@{k}"] = score
        logger.info("  pass@%-4d  %.4f  (%.1f%%)", k, score, score * 100)
    logger.info("=" * 50)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="HumanEval benchmark for SLM")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--samples-file", type=str, default=str(SAMPLES_FILE))
    args = parser.parse_args()

    samples_path = Path(args.samples_file)
    k_values = [k for k in [1, 10, 100] if k <= args.n_samples] or [1]

    problems = load_humaneval_problems()

    if not args.eval_only:
        logger.info("=== HumanEval Generation ===")
        logger.info("n_samples   : %d", args.n_samples)
        logger.info("temperature : %.2f", args.temperature)
        logger.info("checkpoint  : %s", args.checkpoint)

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
    results = run_evaluation(samples_path, problems, k_values)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "benchmark": "HumanEval",
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass_at_k": results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
