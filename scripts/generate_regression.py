"""
scripts/generate_regression.py

Generates the regression test fixture for the tokenizer.
Run this once after training, then commit the output.
Re-run only when you intentionally retrain the tokenizer.

Usage:
    python scripts/generate_regression.py
"""

import json
from pathlib import Path

from tokenizer.tokenizer import SLMTokenizer

TOKENIZER_DIR = Path("tokenizer/trained")
OUTPUT_FILE = Path("tests/unit/tokenizer_regression.json")

REGRESSION_TEXTS = [
    "def foo(): pass",
    "import os\nimport sys\n",
    "class MyModel(nn.Module):\n    pass\n",
    "x = {'key': [1, 2, 3]}\n",
    "    for i in range(10):\n        print(i)\n",
    "def __init__(self, hidden_dim: int) -> None:\n    super().__init__()\n",
    "return torch.nn.functional.softmax(x, dim=-1)\n",
    "if __name__ == '__main__':\n    main()\n",
    "# comment\nx: int = 42\n",
    "async def fetch(url: str) -> dict:\n    pass\n",
]


def main():
    tok = SLMTokenizer(TOKENIZER_DIR)
    cases = []
    for text in REGRESSION_TEXTS:
        ids = tok.encode(text)
        cases.append({"text": text, "ids": ids})
        print(f"  {repr(text[:40]):45s} → {len(ids)} tokens")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\nRegression fixture saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
