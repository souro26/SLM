"""
tokenizer/tokenizer.py

Wrapper around the trained BPE tokenizer.
This is the only file the rest of the project imports for tokenization.

Usage:
    from tokenizer.tokenizer import SLMTokenizer

    tok = SLMTokenizer("tokenizer/trained")
    ids = tok.encode("def foo(x):")
    text = tok.decode(ids)
"""

import json
from pathlib import Path

from tokenizers import Tokenizer


class SLMTokenizer:
    def __init__(self, tokenizer_dir: str | Path):
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_path = tokenizer_dir / "tokenizer.json"
        meta_path = tokenizer_dir / "tokenizer_meta.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found at {tokenizer_dir}. " "Run tokenizer/train.py first."
            )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            special = meta.get("special_tokens", {})
        else:
            special = {}

        self._eof_id: int = special.get(
            "<|endoffile|>", self._tokenizer.token_to_id("<|endoffile|>")
        )
        self._pad_id: int = special.get("<|pad|>", self._tokenizer.token_to_id("<|pad|>"))
        self._unk_id: int = special.get("<|unk|>", self._tokenizer.token_to_id("<|unk|>"))

        self._tokenizer.no_padding()

    def encode(self, text: str, add_eof: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text).ids
        if add_eof:
            ids.append(self._eof_id)
        return ids

    def encode_batch(self, texts: list[str], add_eof: bool = False) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(texts)
        ids_list = [enc.ids for enc in encodings]
        if add_eof:
            ids_list = [ids + [self._eof_id] for ids in ids_list]
        return ids_list

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, ids_list: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        return self._tokenizer.decode_batch(ids_list, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def eof_id(self) -> int:
        return self._eof_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._tokenizer.id_to_token(id)

    def get_vocab(self) -> dict:
        return self._tokenizer.get_vocab()

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"SLMTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"eof_id={self.eof_id}, "
            f"pad_id={self.pad_id}, "
            f"unk_id={self.unk_id})"
        )
