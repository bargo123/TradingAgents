from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ContractError, SequenceTooLongError
from .models import SFTExample
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
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length < 1:
            raise ContractError("max_length must be positive")
        native = getattr(tok, "chat_template", None)
        if not native and self.template != "fallback-v1":
            raise ContractError("native chat template unavailable; explicitly select fallback-v1")
        model_max = getattr(tok, "model_max_length", self.max_length)
        if model_max is None:
            model_max = self.max_length
        if isinstance(model_max, bool) or not isinstance(model_max, int) or model_max <= 0:
            raise ContractError("tokenizer model_max_length is invalid")
        return tok

    @staticmethod
    def effective_max_length(tokenizer: Any, configured: int) -> int:
        """Return the actual capacity used for this tokenizer/configuration."""
        if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
            raise ContractError("max_length must be positive")
        model_max = getattr(tokenizer, "model_max_length", configured)
        # Transformers uses a very large sentinel when a tokenizer has no
        # declared model limit; the explicit Phase 11 setting is authoritative
        # in that case.
        if model_max is None:
            model_max = configured
        if isinstance(model_max, bool) or not isinstance(model_max, int) or model_max <= 0:
            raise ContractError("tokenizer model_max_length is invalid")
        if model_max > 10_000_000:
            model_max = configured
        return min(configured, model_max)

    def fingerprint(self) -> dict[str, Any]:
        tok = self.tokenizer
        config: dict[str, Any] = {}
        if tok is not None:
            for name in (
                "model_max_length", "padding_side", "truncation_side", "padding",
                "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
            ):
                value = getattr(tok, name, None)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    config[name] = value
            vocab = getattr(tok, "get_vocab", None)
            if callable(vocab):
                try:
                    entries = vocab()
                    if isinstance(entries, dict):
                        config["vocab"] = {
                            str(key): int(value)
                            for key, value in sorted(entries.items(), key=lambda item: str(item[0]))
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                except (TypeError, ValueError):
                    pass
        # Vocabulary entries are ordinary model metadata; words such as
        # ``prompt`` or ``reasoning`` are not hidden runtime content.  Hash
        # this bounded primitive-only tokenizer description directly instead
        # of routing it through the training-data sensitive-field scanner.
        raw = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        template = getattr(tok, "chat_template", None) if tok is not None else None
        template_text = template if isinstance(template, str) else repr(template)
        return {
            "policy_version": self.policy_version,
            "template": self.template or "native",
            "tokenizer": getattr(tok, "name_or_path", "lazy"),
            "model_max_length": getattr(tok, "model_max_length", None),
            "effective_max_length": self.effective_max_length(tok, self.max_length) if tok is not None else self.max_length,
            "chat_template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
            "config_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }


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
    try:
        prefix_rendered = tok.apply_chat_template(
            list(messages[:-1]), tokenize=False, add_generation_prompt=True, truncation=False
        ) if getattr(tok, "chat_template", None) else _render(tok, messages[:-1], policy.template)
    except TypeError:
        prefix_rendered = _render(tok, messages[:-1], policy.template)
    boundary = _encode(tok, prefix_rendered)
    labels = [-100] * min(len(boundary), len(ids)) + list(ids[min(len(boundary), len(ids)):])
    effective_max = policy.effective_max_length(tok, policy.max_length)
    if len(ids) > effective_max:
        candidate, reduction = reduce_context(
            messages,
            effective_max,
            lambda value: len(_encode(tok, _render(tok, value, policy.template))),
        )
        if candidate != messages:
            rendered = _render(tok, candidate, policy.template)
            ids = _encode(tok, rendered)
            try:
                prefix_rendered = tok.apply_chat_template(
                    list(candidate[:-1]), tokenize=False, add_generation_prompt=True, truncation=False
                ) if getattr(tok, "chat_template", None) else _render(tok, candidate[:-1], policy.template)
            except TypeError:
                prefix_rendered = _render(tok, candidate[:-1], policy.template)
            boundary = _encode(tok, prefix_rendered)
            labels = [-100] * min(len(boundary), len(ids)) + list(ids[min(len(boundary), len(ids)):])
        else:
            raise SequenceTooLongError(f"sequence length {len(ids)} exceeds maximum {effective_max}")
    if len(ids) > effective_max:
        raise SequenceTooLongError(f"sequence length {len(ids)} exceeds maximum {effective_max}")
    return TokenizedExample(ids, tuple(labels), tuple([1] * len(ids)), reduction if candidate != messages else None)


__all__ = ["TokenizationPolicy", "TokenizedExample", "tokenize_example"]
