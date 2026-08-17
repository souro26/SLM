"""
eval/ds1000.py

DS-1000 (Data Science 1000) benchmark evaluation for the SLM.

1,000 realistic data science problems across 7 core Python libraries:
Pandas, NumPy, Matplotlib, SciPy, Scikit-learn, PyTorch, and Statsmodels.

Requirements:
    pip install datasets

Usage:
    # Generate 1 sample per problem (greedy or t=0.2)
    python -m eval.ds1000 --n-samples 1 --temperature 0.0

    # Specific library only (e.g. numpy or pandas)
    python -m eval.ds1000 --lib Pandas

    # Evaluate existing samples
    python -m eval.ds1000 --eval-only

Output:
    results/ds1000_samples.jsonl — raw completions
    results/ds1000_results.json  — accuracy breakdown per library and total
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
SAMPLES_FILE = RESULTS_DIR / "ds1000_samples.jsonl"
RESULTS_FILE = RESULTS_DIR / "ds1000_results.json"


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "eval_ds1000.log"
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

# Stop at double newline or next function/top-level block
STOP_STRINGS = ["\n# ---", "\ndef ", "\nclass ", "\nif __name__"]
TEMPERATURE = 0.2
TOP_P = 0.95
EXEC_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_ds1000_problems(library_filter: str | None = None) -> list[dict]:
    """Load DS-1000 dataset from HuggingFace cache or direct download."""
    logger.info("Loading DS-1000 dataset...")
    cache_path = LOGS_DIR / "ds1000_test.jsonl"
    if not cache_path.exists():
        url = "https://huggingface.co/datasets/xlangai/DS-1000/resolve/main/test.jsonl"
        logger.info("Downloading DS-1000 dataset from HuggingFace (%s)...", url)
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(cache_path, "wb") as out_file:
            out_file.write(resp.read())
        logger.info("Downloaded DS-1000 dataset to %s", cache_path)

    problems = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            lib = row.get("lib", row.get("library", "Unknown"))
            if library_filter and lib.lower() != library_filter.lower():
                continue
            problems.append(
                {
                    "id": row.get("id", row.get("metadata", {}).get("problem_id")),
                    "prompt": row["prompt"],
                    "reference_code": row.get("reference_code", ""),
                    "test_code": row.get("code_context", row.get("test", "")),
                    "lib": lib,
                }
            )

    logger.info("Loaded %d DS-1000 problems", len(problems))
    return problems


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
        raw_prompt = problem["prompt"]
        pid = problem["id"]

        # DS-1000 prompts contain a [insert] placeholder.
        # We prompt the model with everything BEFORE [insert].
        model_prompt = raw_prompt.split("[insert]")[0] if "[insert]" in raw_prompt else raw_prompt

        logger.info(
            "[%d/%d] id=%s (%s) — generating %d sample(s)...",
            i + 1,
            total,
            pid,
            problem["lib"],
            n_samples,
        )
        t0 = time.time()

        completions = gen.generate(
            prompt=model_prompt,
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
                    "id": pid,
                    "lib": problem["lib"],
                    "raw_prompt": raw_prompt,
                    "model_prompt": model_prompt,
                    "completion": completion.lstrip(" "),
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
# Code Execution & Evaluation
# ---------------------------------------------------------------------------


def _run_code(code: str, timeout: float) -> bool:
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return res.returncode == 0
    except Exception:
        return False


def evaluate_samples(
    samples_path: Path, problems: list[dict], timeout: float = EXEC_TIMEOUT
) -> dict:
    """Evaluate generated completions against test cases."""
    samples_by_id: dict[str, list[dict]] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples_by_id.setdefault(str(obj["id"]), []).append(obj)

    prob_map = {str(p["id"]): p for p in problems}

    lib_results: dict[str, list[bool]] = {}
    total_correct = 0
    total_count = 0

    for pid, problem in prob_map.items():
        sample_list = samples_by_id.get(pid, [])
        if not sample_list:
            continue

        lib = problem["lib"]
        test_context = problem["test_code"]

        passed = False
        for s in sample_list:
            raw_prompt = s.get("raw_prompt", s.get("prompt", ""))
            completion = s["completion"]

            if "[insert]" in raw_prompt:
                filled_code = raw_prompt.replace("[insert]", completion)
            else:
                filled_code = raw_prompt + completion

            if test_context:
                if "[insert]" in test_context:
                    code_to_run = test_context.replace("[insert]", completion)
                else:
                    code_to_run = filled_code + "\n" + test_context
            else:
                code_to_run = filled_code

            if _run_code(code_to_run, timeout):
                passed = True
                break

        lib_results.setdefault(lib, []).append(passed)
        if passed:
            total_correct += 1
        total_count += 1

    summary = {}
    logger.info("")
    logger.info("=" * 50)
    for lib, res in sorted(lib_results.items()):
        acc = sum(res) / len(res) if res else 0.0
        summary[lib] = {"correct": sum(res), "total": len(res), "accuracy": acc}
        logger.info("  %-15s : %d/%d (%.1f%%)", lib, sum(res), len(res), acc * 100)

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    summary["overall"] = {"correct": total_correct, "total": total_count, "accuracy": overall_acc}
    logger.info(
        "  %-15s : %d/%d (%.1f%%)", "OVERALL", total_correct, total_count, overall_acc * 100
    )
    logger.info("=" * 50)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="DS-1000 benchmark for SLM")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--lib", type=str, default=None, help="Filter by library (Pandas, Numpy, etc)"
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--samples-file", type=str, default=str(SAMPLES_FILE))
    args = parser.parse_args()

    samples_path = Path(args.samples_file)
    problems = load_ds1000_problems(args.lib)

    if not args.eval_only:
        logger.info("=== DS-1000 Generation ===")
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
        logger.info("Skipping generation, using %s", samples_path)

    logger.info("")
    logger.info("=== DS-1000 Evaluation ===")
    results = evaluate_samples(samples_path, problems)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "benchmark": "DS-1000",
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "results": results,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
