"""
eval/generator.py

Core generation engine shared by all benchmarks.

Loads the trained TransformerModel from a checkpoint directory and the
SLMTokenizer, then exposes a high-level generate() interface:

    - Temperature sampling
    - Top-p (nucleus) filtering
    - Stop-string detection (stops before emitting the stop token sequence)
    - KV-cache for O(1) per-step decoding after the initial prefill
    - Multiple samples per prompt (for pass@k evaluation)

Usage:
    from eval.generator import CodeGenerator

    gen = CodeGenerator(
        checkpoint_dir="checkpoints/pilot-001/step_061000",
        model_config_path="configs/model.yaml",
        tokenizer_dir="tokenizer/trained",
    )

    completions = gen.generate(
        prompt="def binary_search(arr, target):\n    ",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
        num_samples=1,
    )
    print(completions[0])
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as f

from model.config import ModelConfig
from model.transformer import TransformerModel
from tokenizer.tokenizer import SLMTokenizer

logger = logging.getLogger(__name__)

DEFAULT_STOP_STRINGS = ["\ndef ", "\nclass ", "\n# ---", "\nif __name__"]


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out all logits outside the top-p nucleus."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs - f.softmax(sorted_logits, dim=-1) > top_p
    logits_filtered = logits.clone()
    logits_filtered[sorted_indices[sorted_indices_to_remove]] = float("-inf")
    return logits_filtered


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample the next token id from the last-position logits."""
    if temperature == 0.0:
        return int(logits.argmax().item())
    logits = logits / temperature
    logits = _top_p_filter(logits, top_p)
    probs = f.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class CodeGenerator:
    """
    Wraps a trained TransformerModel for autoregressive code generation.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Path to a checkpoint directory containing tensors.pt and metadata.json.
        Typically checkpoints/pilot-001/step_061000.
    model_config_path : str | Path
        Path to configs/model.yaml.
    tokenizer_dir : str | Path
        Path to the trained tokenizer directory (tokenizer/trained/).
    device : str
        "cuda", "cpu", or "auto" (picks cuda if available).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_config_path: str | Path = "configs/model.yaml",
        tokenizer_dir: str | Path = "tokenizer/trained",
        device: str = "auto",
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading model config from %s", model_config_path)
        self.cfg = ModelConfig.from_yaml(str(model_config_path))

        logger.info("Building model architecture...")
        self.model = TransformerModel(self.cfg)

        tensor_path = checkpoint_dir / "tensors.pt"
        logger.info("Loading weights from %s", tensor_path)
        state = torch.load(tensor_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded on %s", self.device)

        logger.info("Loading tokenizer from %s", tokenizer_dir)
        self.tokenizer = SLMTokenizer(tokenizer_dir)
        self._eof_id: int = self.tokenizer._eof_id

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_strings: list[str] | None = None,
        num_samples: int = 1,
    ) -> list[str]:
        """
        Generate `num_samples` completions for the given prompt.

        Returns a list of strings — each is the *completion only* (the prompt
        is not included). Stop strings are included up to (not including) the
        triggering sequence so the caller always gets valid function bodies.

        Parameters
        ----------
        prompt : str
            The text to condition on (e.g. function signature + docstring).
        max_new_tokens : int
            Hard cap on tokens generated per sample.
        temperature : float
            Sampling temperature. 0.0 = greedy, 0.2 = near-greedy, 0.8 = creative.
        top_p : float
            Nucleus filtering probability. 1.0 = disabled.
        stop_strings : list[str] | None
            Generation stops when any of these strings appear in the decoded
            output. Defaults to DEFAULT_STOP_STRINGS.
        num_samples : int
            Number of independent completions to generate.
        """
        if stop_strings is None:
            stop_strings = DEFAULT_STOP_STRINGS

        prompt_ids = self.tokenizer.encode(prompt, add_eof=False)

        # Clamp prompt to context_length-1 to always leave room for generation.
        max_prompt_len = self.cfg.context_length - 1
        if len(prompt_ids) > max_prompt_len:
            logger.warning(
                "Prompt is %d tokens, truncating to %d (context_length-1)",
                len(prompt_ids),
                max_prompt_len,
            )
            prompt_ids = prompt_ids[-max_prompt_len:]  # keep the tail (most recent context)

        # Clamp max_new_tokens so total sequence fits in context window.
        actual_max_new = min(max_new_tokens, self.cfg.context_length - len(prompt_ids))
        if actual_max_new <= 0:
            return [""] * num_samples

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        completions: list[str] = []

        for _ in range(num_samples):
            completion_ids: list[int] = []

            logits, kv_caches = self.model(prompt_tensor)
            next_token = _sample_token(logits[0, -1], temperature, top_p)

            while len(completion_ids) < actual_max_new:
                completion_ids.append(next_token)

                if next_token == self._eof_id:
                    break

                partial = self.tokenizer.decode(completion_ids)
                triggered = False
                for stop in stop_strings:
                    idx = partial.find(stop)
                    if idx != -1:
                        completion_ids = self.tokenizer.encode(partial[:idx], add_eof=False)
                        triggered = True
                        break
                if triggered:
                    break

                next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                logits, kv_caches = self.model(next_tensor, kv_caches=kv_caches)
                next_token = _sample_token(logits[0, -1], temperature, top_p)

            completions.append(self.tokenizer.decode(completion_ids))

        return completions
