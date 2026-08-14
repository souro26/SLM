"""
eval/mbpp.py

MBPP (Mostly Basic Python Programming) benchmark evaluation for the SLM.

374 Python programming problems from Google Research, covering a wider range
of difficulty and topics than HumanEval.

Requirements:
    pip install datasets

Usage:
    # Default: 20 samples per problem, reports pass@1 and pass@10
    python -m eval.mbpp

    # Greedy single sample (fastest, gives pass@1)
    python -m eval.mbpp --n-samples 1 --temperature 0.0

    # Skip generation, just re-run evaluation
    python -m eval.mbpp --eval-only

Output:
    results/mbpp_samples.jsonl   — raw completions
    results/mbpp_results.json    — pass@k scores
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
SAMPLES_FILE = RESULTS_DIR / "mbpp_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / "mbpp_results.json"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "eval_mbpp.log"
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

# Stop at the next top-level definition
STOP_STRINGS = ["\ndef ", "\nclass ", "\nif __name__", "\n# ---"]
TEMPERATURE = 0.8
TOP_P = 0.95

# Execution timeout per test (seconds)
EXEC_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_mbpp_problems() -> list[dict]:
    """Load MBPP test split (374 problems) from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    logger.info("Downloading MBPP dataset from HuggingFace...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test", trust_remote_code=True)
    problems = []
    for row in ds:
        problems.append(
            {
                "task_id": row["task_id"],
                "text": row["text"],
                "code": row["code"],
                "test_list": row["test_list"],
                "test_setup_code": row.get("test_setup_code", ""),
            }
        )
    logger.info("Loaded %d MBPP problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _extract_fn_name(code: str) -> str | None:
    """Extract the first function name from the canonical solution."""
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("def "):
            name = line[4:].split("(")[0].strip()
            if name:
                return name
    return None


def format_mbpp_prompt(problem: dict) -> str:
    """
    Format a MBPP problem as a Python completion prompt.

    The model sees:
        # {problem description}
        # Tests:
        # assert fn(args) == expected
        def fn_name(
    """
    text = problem["text"].strip()
    tests = problem["test_list"]
    code = problem["code"]

    fn_name = _extract_fn_name(code) or "solution"

    lines = [f"# {text}", "# Tests:"]
    for t in tests[:3]:  # show at most 3 test cases as hints
        lines.append(f"# {t.strip()}")
    lines.append(f"def {fn_name}(")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution / pass@k
# ---------------------------------------------------------------------------


def _run_code_in_subprocess(code: str, timeout: float) -> tuple[bool, str]:
    """
    Execute `code` in an isolated subprocess with a timeout.
    Returns (passed: bool, error_message: str).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)


def evaluate_sample(completion: str, problem: dict, timeout: float = EXEC_TIMEOUT) -> bool:
    """
    Test a single completion against MBPP test assertions.

    Builds a script that:
      1. Runs any setup code
      2. Runs the completion (function definition)
      3. Executes each test assertion
    """
    setup = problem.get("test_setup_code", "").strip()
    tests = problem["test_list"]
    prompt = format_mbpp_prompt(problem)

    # Reconstruct the full function from the prompt + completion
    fn_start = prompt.rfind("\ndef ") + 1  # position of "def fn_name("
    prompt_fn_header = prompt[fn_start:]  # "def fn_name("

    # The completion starts after the opening paren in the prompt
    full_fn = prompt_fn_header + completion

    test_code = "\n".join(tests)
    script = "\n\n".join(filter(None, [setup, full_fn, test_code]))

    passed, _ = _run_code_in_subprocess(script, timeout)
    return passed


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator: given n total samples and c correct ones,
    what is the probability that at least 1 of k random picks is correct?

    Formula from the HumanEval paper:
        pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    import math

    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


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
        prompt = format_mbpp_prompt(problem)
        logger.info(
            "[%d/%d] task_id=%s — %d sample(s)...", i + 1, total, problem["task_id"], n_samples
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
                    "task_id": problem["task_id"],
                    "prompt": prompt,
                    "completion": completion,
                }
            )

    return samples


def save_samples(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    samples_path: Path,
    problems: list[dict],
    k_values: list[int],
    timeout: float = EXEC_TIMEOUT,
) -> dict:
    # Load samples
    samples_by_task: dict[int | str, list[str]] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tid = obj["task_id"]
            samples_by_task.setdefault(tid, []).append(obj["completion"])

    problem_map = {p["task_id"]: p for p in problems}

    pass_at_k_accum: dict[int, list[float]] = {k: [] for k in k_values}

    for tid, problem in problem_map.items():
        completions = samples_by_task.get(tid, [])
        n = len(completions)
        if n == 0:
            logger.warning("No samples for task_id=%s, skipping", tid)
            continue

        c = sum(evaluate_sample(comp, problem, timeout) for comp in completions)

        logger.info("  task_id=%-6s  %d/%d passed", tid, c, n)

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
    parser = argparse.ArgumentParser(description="MBPP benchmark for SLM")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--timeout", type=float, default=EXEC_TIMEOUT, help="Per-test execution timeout in seconds"
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--samples-file", type=str, default=str(SAMPLES_FILE))
    args = parser.parse_args()

    samples_path = Path(args.samples_file)
    k_values = [k for k in [1, 10, 100] if k <= args.n_samples] or [1]

    problems = load_mbpp_problems()

    if not args.eval_only:
        logger.info("=== MBPP Generation ===")
        logger.info("n_samples   : %d", args.n_samples)
        logger.info("temperature : %.2f", args.temperature)
        logger.info("checkpoint  : %s", args.checkpoint)
        logger.info("")

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

    logger.info("")
    logger.info("=== MBPP Evaluation (pass@%s) ===", k_values)
    results = run_evaluation(samples_path, problems, k_values, args.timeout)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "benchmark": "MBPP",
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass_at_k": results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
