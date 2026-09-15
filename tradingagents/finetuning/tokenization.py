from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from .errors import ContractError, SequenceTooLongError
from .models import SFTExample, canonical_json
from .sequence import reduce_context


@dataclass(frozen=True, slots=True)
class TokenizationPolicy:
    tokenizer: Any = None
    tokenizer_loader: Callable[[], Any] | None = None
    max_length: int = 4096
    template: str | None = None
    policy_version: str = "phase11-tokenization.v1"

    def load(self):
        tok = self.tokenizer if self.tokenizer is not None else (self.tokenizer_loader() if self.tokenizer_loader else None)
        if tok is None:
            raise ContractError("tokenizer is required")
        if self.max_length < 1:
            raise ContractError("max_length must be positive")
        model_max = getattr(tok, "model_max_length", self.max_length)
        native = getattr(tok, "chat_template", None)
        if not native and self.template != "fallback-v1":
            raise ContractError("native chat template unavailable; explicitly select fallback-v1")
        return tok

    def fingerprint(self) -> dict[str, Any]:
        tok = self.tokenizer
        config = getattr(tok, "init_kwargs", getattr(tok, "config", {})) if tok is not None else {}
        raw = canonical_json(config) if isinstance(config, (dict, list, tuple)) else str(config)
        return {"policy_version": self.policy_version, "template": self.template or "native", "tokenizer": getattr(tok, "name_or_path", "lazy"), "model_max_length": getattr(tok, "model_max_length", None), "config_sha256": hashlib.sha256(raw.encode()).hexdigest()}


@dataclass(frozen=True, slots=True)
class TokenizedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    reduction: dict[str, Any] | None = None


def _render(tok, messages, template):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(list(messages), tokenize=False, truncation=False)
    # Explicit fallback is intentionally simple and deterministic.
    return "".join(f"<{m['role']}> {m['content']}\n" for m in messages)


def _encode(tok, value):
    ids = tok.encode(value, add_special_tokens=False, truncation=False)
    return tuple(int(x) for x in ids)


def tokenize_example(example: SFTExample, policy: TokenizationPolicy) -> TokenizedExample:
    tok = policy.load()
    messages = tuple(dict(m) for m in example.messages)
    candidate = messages
    rendered = _render(tok, messages, policy.template)
    ids = _encode(tok, rendered)
    boundary = _encode(tok, _render(tok, messages[:-1], policy.template))
    labels = [-100] * min(len(boundary), len(ids)) + list(ids[min(len(boundary), len(ids)):])
    effective_max = min(policy.max_length, getattr(tok, "model_max_length", policy.max_length) or policy.max_length)
    if len(ids) > effective_max:
        candidate, reduction = reduce_context(messages, effective_max, len)
        if candidate != messages:
            rendered = _render(tok, candidate, policy.template); ids = _encode(tok, rendered)
            boundary = _encode(tok, _render(tok, candidate[:-1], policy.template))
            labels = [-100] * min(len(boundary), len(ids)) + list(ids[min(len(boundary), len(ids)):])
        else:
            raise SequenceTooLongError(f"sequence length {len(ids)} exceeds maximum {policy.max_length}")
    if len(ids) > effective_max:
        raise SequenceTooLongError(f"sequence length {len(ids)} exceeds maximum {effective_max}")
    return TokenizedExample(ids, tuple(labels), tuple([1] * len(ids)), reduction if candidate != messages else None)


__all__ = ["TokenizationPolicy", "TokenizedExample", "tokenize_example"]
