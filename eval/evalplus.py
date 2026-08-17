"""
eval/evalplus.py

HumanEval+ and MBPP+ evaluation via the EvalPlus framework.

EvalPlus enhances the original HumanEval (164 problems) and MBPP (374 problems)
with far more rigorous test cases — many models that score well on the originals
drop 10-30% here because the extra tests catch edge cases.

Requirements:
    pip install evalplus

Usage:
    # HumanEval+ (20 samples → pass@1 and pass@10)
    python -m eval.evalplus --dataset humaneval

    # MBPP+
    python -m eval.evalplus --dataset mbpp

    # Both
    python -m eval.evalplus --dataset humaneval mbpp

    # Skip generation, just evaluate existing samples
    python -m eval.evalplus --dataset humaneval --eval-only

Evaluation is delegated to the official EvalPlus CLI after generation:
    python -m evalplus.evaluate --dataset humaneval --samples results/evalplus_humaneval_samples.jsonl
    python -m evalplus.evaluate --dataset mbpp       --samples results/evalplus_mbpp_samples.jsonl
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


def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "eval_evalplus.log"
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

DATASET_CHOICES = ["humaneval", "mbpp"]


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
# Dataset loading via EvalPlus
# ---------------------------------------------------------------------------


def load_evalplus_problems(dataset: str) -> dict[str, dict]:
    """
    Load HumanEval+ or MBPP+ problems using the evalplus package.

    Returns {task_id: {task_id, prompt, ...}}.
    """
    try:
        import evalplus.data as epdata
    except ImportError:
        logger.error("evalplus not installed. Run:\n" "  pip install evalplus")
        sys.exit(1)

    if dataset == "humaneval":
        problems = epdata.get_human_eval_plus()
    elif dataset == "mbpp":
        problems = epdata.get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info("Loaded %d %s+ problems", len(problems), dataset)
    return problems


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
        prompt = _ensure_indented_prompt(problem["prompt"])
        logger.info("[%d/%d] %s — %d sample(s)...", i + 1, total, task_id, n_samples)
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
            samples.append({"task_id": task_id, "completion": completion.lstrip(" ")})

    return samples


def save_samples(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


# ---------------------------------------------------------------------------
# Evaluation via EvalPlus CLI
# ---------------------------------------------------------------------------


def run_evalplus_evaluation(dataset: str, samples_path: Path) -> None:
    """
    Invoke the official EvalPlus evaluator as a subprocess.

    This runs the rigorous test suite (80x more tests than the original
    HumanEval/MBPP) and prints pass@k directly.
    """
    import os

    env = os.environ.copy()
    # Add eval/compat directory to PYTHONPATH so dummy 'resource' module is found on Windows
    # without shadowing the installed 'evalplus' package.
    compat_dir = str(Path(__file__).resolve().parent / "compat")
    env["PYTHONPATH"] = f"{compat_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        str(samples_path),
    ]
    logger.info("Running EvalPlus evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.error("EvalPlus evaluation exited with code %d", result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="EvalPlus benchmark (HumanEval+ / MBPP+)")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["humaneval"],
        choices=DATASET_CHOICES,
        help="Which dataset(s) to run: humaneval, mbpp, or both",
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip generation; evaluate existing samples files"
    )
    args = parser.parse_args()

    for dataset in args.dataset:
        samples_path = RESULTS_DIR / f"evalplus_{dataset}_samples.jsonl"

        logger.info("\n=== EvalPlus: %s+ ===", dataset.upper())

        if not args.eval_only:
            logger.info("n_samples   : %d", args.n_samples)
            logger.info("temperature : %.2f", args.temperature)
            logger.info("checkpoint  : %s", args.checkpoint)

            problems = load_evalplus_problems(dataset)
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
        run_evalplus_evaluation(dataset, samples_path)


if __name__ == "__main__":
    main()
