"""
Resilient raw tokenizer.json wrapper (zero PyTorch / transformers required).
"""
from __future__ import annotations

from pathlib import Path
from .templates import format_messages


class ResilientTokenizer:
    def __init__(self, model_dir: str | Path, template: str = "qwen35"):
        p = Path(model_dir).expanduser()
        self.template = template
        self.eos_ids = {248046, 248044, 151645, 151643, 128009, 128001}
        tj = p / "tokenizer.json"

        if tj.exists():
            try:
                from tokenizers import Tokenizer
                self._tok = Tokenizer.from_file(str(tj))
                self._mode = "fast_raw"
                for tok_str in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
                    tid = self._tok.token_to_id(tok_str)
                    if tid is not None:
                        self.eos_ids.add(int(tid))
                return
            except ImportError as e:
                raise ImportError(
                    "Missing `tokenizers` package. Install the only Python dependency with:\n"
                    "  pip3 install tokenizers"
                ) from e

        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(str(p))
        self._mode = "hf"
        if getattr(self._tok, "eos_token_id", None) is not None:
            self.eos_ids.add(int(self._tok.eos_token_id))

    def encode_raw(self, text: str) -> list[int]:
        if not text:
            return []
        if self._mode == "fast_raw":
            return self._tok.encode(text, add_special_tokens=False).ids
        return self._tok.encode(text, add_special_tokens=False)

    def format_chat_with_prefix(self, messages: list[dict], enable_thinking: bool = False) -> tuple[list[int], int]:
        sys_str, full_str = format_messages(messages, template=self.template, enable_thinking=enable_thinking)
        full_ids = self.encode_raw(full_str)
        sys_ids = self.encode_raw(sys_str) if sys_str else []
        prefix_len = len(sys_ids) if len(sys_ids) >= 16 else 0
        return full_ids, prefix_len

    def token_id(self, text: str) -> int | None:
        try:
            tid = self._tok.token_to_id(text) if self._mode == "fast_raw" else self._tok.convert_tokens_to_ids(text)
        except Exception:
            return None
        return int(tid) if tid is not None and tid >= 0 else None

    def decode(self, ids: list[int]) -> str:
        clean = [t for t in ids if t not in self.eos_ids]
        return self._tok.decode(clean, skip_special_tokens=True)
